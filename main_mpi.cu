#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "perception_common.h"
#include "perception_cpu.h"
#include "perception_gpu.h"
#include "perception_sgm.h"

struct MpiRunOptions {
    StereoParams p;
    bool         verify        = true;
    bool         save_images   = false;
    std::string  csv_path      = "";
    std::string  left_path     = "";
    std::string  right_path    = "";
    std::string  output_prefix = "disp_mpi";
    bool         use_sgm       = false;   // distributed 4-path SGM instead of SAD
    bool         sgm_census    = false;
    int          p1            = 200;
    int          p2            = 500;
    bool         p1_set        = false;
    bool         p2_set        = false;
};

static void parse_args(int argc, char** argv, MpiRunOptions& opt) {
    StereoParams& p = opt.p;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if ((key == "--height"   || key == "-H") && i+1<argc) { p.height    = std::stoi(argv[++i]); continue; }
        if ((key == "--width"    || key == "-W") && i+1<argc) { p.width     = std::stoi(argv[++i]); continue; }
        if ((key == "--disp"     || key == "-d") && i+1<argc) { p.true_disp = std::stoi(argv[++i]); continue; }
        if ((key == "--max-disp" || key == "-m") && i+1<argc) { p.max_disp  = std::stoi(argv[++i]); continue; }
        if ((key == "--radius"   || key == "-r") && i+1<argc) { p.radius    = std::stoi(argv[++i]); continue; }
        if ((key == "--repeats"  || key == "-n") && i+1<argc) { p.repeats   = std::stoi(argv[++i]); continue; }
        if (key == "--no-verify")                              { opt.verify        = false;       continue; }
        if (key == "--save-images")                            { opt.save_images   = true;        continue; }
        if (key == "--csv"            && i+1<argc)             { opt.csv_path      = argv[++i];   continue; }
        if (key == "--left"           && i+1<argc)             { opt.left_path     = argv[++i];   continue; }
        if (key == "--right"          && i+1<argc)             { opt.right_path    = argv[++i];   continue; }
        if (key == "--output-prefix"  && i+1<argc)             { opt.output_prefix = argv[++i];   continue; }
        if (key == "--sgm")                                    { opt.use_sgm       = true;        continue; }
        if (key == "--sgm-census")                             { opt.sgm_census    = true;        continue; }
        if (key == "--p1"             && i+1<argc)             { opt.p1 = std::stoi(argv[++i]); opt.p1_set = true; continue; }
        if (key == "--p2"             && i+1<argc)             { opt.p2 = std::stoi(argv[++i]); opt.p2_set = true; continue; }
    }
    // Census Hamming costs are much smaller than window-SAD; use census defaults.
    if (opt.sgm_census && !opt.p1_set) opt.p1 = 7;
    if (opt.sgm_census && !opt.p2_set) opt.p2 = 42;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0)
        cudaSetDevice(rank % num_devices);

    MpiRunOptions opt;
    parse_args(argc, argv, opt);
    const StereoParams& p = opt.p;
    bool using_real = !opt.left_path.empty();

    int bcast_dims[2] = {p.height, p.width};
    if (rank == 0 && using_real) {
        Image tmp_probe = load_pgm(opt.left_path);
        bcast_dims[0] = tmp_probe.height;
        bcast_dims[1] = tmp_probe.width;
    }
    MPI_Bcast(bcast_dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    const int H = bcast_dims[0];
    const int W = bcast_dims[1];

    if (nranks > H) {
        if (rank == 0)
            std::cerr << "Error: more MPI ranks (" << nranks
                      << ") than image rows (" << H << ")\n";
        MPI_Finalize();
        return 1;
    }

    // Only rank 0 needs the full image buffers.
    std::vector<uint8_t> full_left(rank == 0 ? H * W : 0);
    std::vector<uint8_t> full_right(rank == 0 ? H * W : 0);
    if (rank == 0) {
        if (using_real) {
            Image tmp_l = load_pgm(opt.left_path);
            Image tmp_r = load_pgm(opt.right_path);
            CHECK(tmp_l.height == H && tmp_l.width == W, "left/right PGM size mismatch");
            CHECK(tmp_r.height == H && tmp_r.width == W, "right PGM size does not match left");
            std::copy(tmp_l.data, tmp_l.data + H * W, full_left.data());
            std::copy(tmp_r.data, tmp_r.data + H * W, full_right.data());
        } else {
            Image tmp_l(H, W), tmp_r(H, W);
            generate_left_image(tmp_l);
            generate_right_image(tmp_l, tmp_r, p.true_disp);
            std::copy(tmp_l.data, tmp_l.data + H * W, full_left.data());
            std::copy(tmp_r.data, tmp_r.data + H * W, full_right.data());
        }
    }

    // Row decomposition
    int base_rows  = H / nranks;
    int remainder  = H % nranks;
    int row_start  = rank * base_rows + std::min(rank, remainder);
    int local_rows = base_rows + (rank < remainder ? 1 : 0);
    int row_end    = row_start + local_rows;

    // Halo rows needed above/below the owned region (clamped at image boundary)
    int top_halo_rows = std::min(p.radius, row_start);
    int bot_halo_rows = std::min(p.radius, H - row_end);
    int halo_top      = top_halo_rows;
    int slab_H        = top_halo_rows + local_rows + bot_halo_rows;

    // Scatter: send only each rank's owned rows (no halo), rank 0 is root
    std::vector<int> sc_counts(nranks), sc_displs(nranks);
    sc_displs[0] = 0;
    for (int i = 0; i < nranks; ++i) {
        int lh = base_rows + (i < remainder ? 1 : 0);
        sc_counts[i] = lh * W;
        if (i > 0) sc_displs[i] = sc_displs[i-1] + sc_counts[i-1];
    }

    Image slab_l(slab_H, W), slab_r(slab_H, W);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0_scatter = MPI_Wtime();
    // Scatter owned rows into the middle of each rank's slab (leaving halo slots empty).
    MPI_Scatterv(rank == 0 ? full_left.data()  : nullptr,
                 sc_counts.data(), sc_displs.data(), MPI_UINT8_T,
                 slab_l.data + top_halo_rows * W, local_rows * W, MPI_UINT8_T,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? full_right.data() : nullptr,
                 sc_counts.data(), sc_displs.data(), MPI_UINT8_T,
                 slab_r.data + top_halo_rows * W, local_rows * W, MPI_UINT8_T,
                 0, MPI_COMM_WORLD);
    double t_scatter = MPI_Wtime() - t0_scatter;

    // Halo exchange: each rank exchanges its boundary rows with its neighbors.
    // TAG_DOWN: data flowing toward higher row indices (rank r -> rank r+1)
    // TAG_UP:   data flowing toward lower  row indices (rank r -> rank r-1)
    const int TAG_DOWN = 0, TAG_UP = 1;

    double t0_halo = MPI_Wtime();
    for (uint8_t* slab : {slab_l.data, slab_r.data}) {
        // Send our first owned rows up to rank-1 (their bottom halo);
        // receive rank-1's last owned rows as our top halo.
        if (rank > 0) {
            MPI_Sendrecv(slab + top_halo_rows * W,  top_halo_rows * W, MPI_UINT8_T,
                         rank - 1, TAG_UP,
                         slab,                        top_halo_rows * W, MPI_UINT8_T,
                         rank - 1, TAG_DOWN,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Send our last owned rows down to rank+1 (their top halo);
        // receive rank+1's first owned rows as our bottom halo.
        if (rank < nranks - 1) {
            MPI_Sendrecv(slab + (top_halo_rows + local_rows - bot_halo_rows) * W,
                         bot_halo_rows * W, MPI_UINT8_T, rank + 1, TAG_DOWN,
                         slab + (top_halo_rows + local_rows) * W,
                         bot_halo_rows * W, MPI_UINT8_T, rank + 1, TAG_UP,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    double t_halo = MPI_Wtime() - t0_halo;

    DisparityMap slab_disp(slab_H, W);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0_kernel = MPI_Wtime();
    float local_gpu_ms = 0.0f;

    if (opt.use_sgm) {
        // Distributed 4-path SGM. Horizontal paths are local; vertical paths
        // are pipelined across ranks via frontier (W*D ints) exchange.
        const int D = p.max_disp;
        std::vector<int> fin((size_t)W * D), fout((size_t)W * D);
        const int TAG_TB = 10, TAG_BT = 11;

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);

        void* ctx = sgm_dist_begin(slab_l, slab_r, p.max_disp, p.radius,
                                   opt.p1, opt.p2, opt.sgm_census ? 1 : 0,
                                   halo_top, local_rows);

        // Top -> bottom sweep: receive frontier from rank-1, send to rank+1.
        if (rank > 0)
            MPI_Recv(fin.data(), W * D, MPI_INT, rank - 1, TAG_TB,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sgm_dist_vert(ctx, +1, rank > 0 ? fin.data() : nullptr, fout.data());
        if (rank < nranks - 1)
            MPI_Send(fout.data(), W * D, MPI_INT, rank + 1, TAG_TB, MPI_COMM_WORLD);

        // Bottom -> top sweep: receive frontier from rank+1, send to rank-1.
        if (rank < nranks - 1)
            MPI_Recv(fin.data(), W * D, MPI_INT, rank + 1, TAG_BT,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sgm_dist_vert(ctx, -1, rank < nranks - 1 ? fin.data() : nullptr, fout.data());
        if (rank > 0)
            MPI_Send(fout.data(), W * D, MPI_INT, rank - 1, TAG_BT, MPI_COMM_WORLD);

        sgm_dist_finish(ctx, slab_disp);

        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&local_gpu_ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    } else {
        local_gpu_ms = sad_stereo_gpu_tiled(slab_l, slab_r, slab_disp,
                                            p.max_disp, p.radius, p.repeats);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_kernel_wall = MPI_Wtime() - t0_kernel;

    float max_gpu_ms = 0.0f;
    MPI_Reduce(&local_gpu_ms, &max_gpu_ms, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> sendcounts(nranks), displs(nranks);
    displs[0] = 0;
    for (int i = 0; i < nranks; ++i) {
        int lh = base_rows + (i < remainder ? 1 : 0);
        sendcounts[i] = lh * W;
        if (i > 0) displs[i] = displs[i-1] + sendcounts[i-1];
    }

    std::vector<disp_t> full_disp(rank == 0 ? H * W : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0_gather = MPI_Wtime();
    MPI_Gatherv(slab_disp.data + halo_top * W,
                local_rows * W, MPI_FLOAT,
                full_disp.data(), sendcounts.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);
    double t_gather = MPI_Wtime() - t0_gather;

    if (rank == 0) {
        double t_comm_total = (t_scatter + t_halo + t_gather) * 1e3;  // ms
        double t_wall_ms    = t_kernel_wall * 1e3;

        const char* algo = opt.use_sgm
            ? (opt.sgm_census ? "SGM 4-path (Census)" : "SGM 4-path (SAD)")
            : "SAD (tiled)";
        std::cout << "\n====================================================\n";
        std::cout << "  MPI+CUDA Stereo " << algo << "  (nranks=" << nranks << ")\n";
        std::cout << "====================================================\n";
        std::cout << "  Image          : " << H << " x " << W << "\n";
        std::cout << "  Max disp/radius: " << p.max_disp << " / " << p.radius << "\n";
        std::cout << "  Rows per rank  : ~" << base_rows
                  << (remainder ? " (some ranks get +1)" : "") << "\n";
        std::cout << "  GPUs visible   : " << num_devices << "\n";
        std::cout << "----------------------------------------------------\n";
        std::cout << "  Scatter (input): " << t_scatter * 1e3 << " ms\n";
        std::cout << "  Halo exchange  : " << t_halo    * 1e3 << " ms\n";
        std::cout << "  GPU kernel max : " << max_gpu_ms       << " ms\n";
        std::cout << "  Kernel wall    : " << t_wall_ms        << " ms  (barrier-to-barrier)\n";
        std::cout << "  Gather (disp)  : " << t_gather  * 1e3 << " ms\n";
        std::cout << "  Total comm     : " << t_comm_total     << " ms\n";
        std::cout << "  Comm fraction  : "
                  << 100.0 * t_comm_total / (t_comm_total + t_wall_ms) << " %\n";
        std::cout << "----------------------------------------------------\n";

        if (opt.verify && !using_real) {
            DisparityMap gt(H, W), combined(H, W);
            generate_ground_truth(gt, p.true_disp, p.radius, p.max_disp);
            std::copy(full_disp.begin(), full_disp.end(), combined.data);
            print_accuracy(combined, gt);
        }

        if (opt.save_images) {
            DisparityMap combined(H, W);
            std::copy(full_disp.begin(), full_disp.end(), combined.data);
            std::string pgm_path = opt.output_prefix + ".pgm";
            std::string ppm_path = opt.output_prefix + ".ppm";
            save_disparity_pgm(combined, pgm_path, p.max_disp);
            save_disparity_ppm(combined, ppm_path,  p.max_disp);
            std::cout << "  Saved: " << pgm_path << ", " << ppm_path << "\n";
            if (using_real) {
                Image img_l(H, W), img_r(H, W);
                std::copy(full_left.begin(),  full_left.end(),  img_l.data);
                std::copy(full_right.begin(), full_right.end(), img_r.data);
                save_pgm(img_l, opt.output_prefix + "_left.pgm");
                save_pgm(img_r, opt.output_prefix + "_right.pgm");
                std::cout << "  Saved: " << opt.output_prefix << "_left.pgm, "
                          << opt.output_prefix << "_right.pgm\n";
            }
        }
        std::cout << "====================================================\n\n";

        if (!opt.csv_path.empty()) {
            std::ifstream fin(opt.csv_path);
            bool new_file = !fin.good();
            fin.close();
            std::ofstream csv(opt.csv_path, std::ios::app);
            if (new_file)
                csv << "nranks,height,width,max_disp,radius,kernel_ms,scatter_ms,"
                       "halo_ms,gather_ms,comm_ms,wall_ms\n";
            csv << nranks << "," << H << "," << W << ","
                << p.max_disp << "," << p.radius << ","
                << max_gpu_ms << ","
                << t_scatter * 1e3 << ","
                << t_halo    * 1e3 << ","
                << t_gather  * 1e3 << ","
                << t_comm_total << ","
                << t_wall_ms << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
