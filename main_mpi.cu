// MPI + CUDA driver for distributed stereo SAD disparity estimation.
// Each MPI rank owns a horizontal slab of rows and runs the tiled GPU kernel.
// Input images are broadcast from rank 0; results are gathered back.
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

struct MpiRunOptions {
    StereoParams p;
    bool         verify     = true;
    std::string  csv_path   = "";
    std::string  left_path  = "";
    std::string  right_path = "";
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
        if (key == "--no-verify")                              { opt.verify     = false;      continue; }
        if (key == "--csv"      && i+1<argc)                   { opt.csv_path  = argv[++i]; continue; }
        if (key == "--left"     && i+1<argc)                   { opt.left_path  = argv[++i]; continue; }
        if (key == "--right"    && i+1<argc)                   { opt.right_path = argv[++i]; continue; }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Bind each rank to its own GPU (wraps around if nranks > num_devices)
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0)
        cudaSetDevice(rank % num_devices);

    MpiRunOptions opt;
    parse_args(argc, argv, opt);
    const StereoParams& p = opt.p;
    bool using_real = !opt.left_path.empty();

    // Rank 0 determines actual image dimensions (from file or params), then broadcasts.
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

    std::vector<uint8_t> full_left(H * W, 0), full_right(H * W, 0);
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

    MPI_Barrier(MPI_COMM_WORLD);
    double t0_bcast = MPI_Wtime();
    MPI_Bcast(full_left.data(),  H * W, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(full_right.data(), H * W, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    double t_bcast = MPI_Wtime() - t0_bcast;

    // ---- 1-D row decomposition ----
    // Distribute H rows as evenly as possible; first `remainder` ranks get +1 row.
    int base_rows  = H / nranks;
    int remainder  = H % nranks;
    int row_start  = rank * base_rows + std::min(rank, remainder);
    int local_rows = base_rows + (rank < remainder ? 1 : 0);
    int row_end    = row_start + local_rows;

    // Halo: each rank needs `radius` rows above and below its slab so the SAD
    // patch never reads out-of-bounds.  The kernel's at_border check then
    // correctly marks those halo rows as 0; we do not gather them.
    int halo_start = std::max(0, row_start - p.radius);
    int halo_end   = std::min(H, row_end   + p.radius);
    int halo_top   = row_start - halo_start;  // local index of first owned row
    int slab_H     = halo_end - halo_start;

    // Extract slab (with halos) from the broadcast buffer — no extra MPI needed.
    Image slab_l(slab_H, W), slab_r(slab_H, W);
    std::copy(full_left.data()  + halo_start * W,
              full_left.data()  + halo_end   * W,
              slab_l.data);
    std::copy(full_right.data() + halo_start * W,
              full_right.data() + halo_end   * W,
              slab_r.data);

    // ---- GPU kernel on local slab ----
    DisparityMap slab_disp(slab_H, W);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0_kernel = MPI_Wtime();

    float local_gpu_ms = sad_stereo_gpu_tiled(slab_l, slab_r, slab_disp,
                                               p.max_disp, p.radius, p.repeats);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_kernel_wall = MPI_Wtime() - t0_kernel;

    // Maximum kernel time across ranks (the actual bottleneck)
    float max_gpu_ms = 0.0f;
    MPI_Reduce(&local_gpu_ms, &max_gpu_ms, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- Gather disparity slabs to rank 0 ----
    // Build non-overlapping send counts and displacements (in MPI_INT units).
    std::vector<int> sendcounts(nranks), displs(nranks);
    displs[0] = 0;
    for (int i = 0; i < nranks; ++i) {
        int lh = base_rows + (i < remainder ? 1 : 0);
        sendcounts[i] = lh * W;
        if (i > 0) displs[i] = displs[i-1] + sendcounts[i-1];
    }

    std::vector<int> full_disp(rank == 0 ? H * W : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0_gather = MPI_Wtime();
    MPI_Gatherv(slab_disp.data + halo_top * W,
                local_rows * W, MPI_INT,
                full_disp.data(), sendcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    double t_gather = MPI_Wtime() - t0_gather;

    // ---- Report on rank 0 ----
    if (rank == 0) {
        double t_comm_total = (t_bcast + t_gather) * 1e3;  // ms
        double t_wall_ms    = t_kernel_wall * 1e3;

        std::cout << "\n====================================================\n";
        std::cout << "  MPI+CUDA Stereo SAD  (nranks=" << nranks << ")\n";
        std::cout << "====================================================\n";
        std::cout << "  Image          : " << H << " x " << W << "\n";
        std::cout << "  Max disp/radius: " << p.max_disp << " / " << p.radius << "\n";
        std::cout << "  Rows per rank  : ~" << base_rows
                  << (remainder ? " (some ranks get +1)" : "") << "\n";
        std::cout << "  GPUs visible   : " << num_devices << "\n";
        std::cout << "----------------------------------------------------\n";
        std::cout << "  Bcast (input)  : " << t_bcast  * 1e3 << " ms\n";
        std::cout << "  GPU kernel max : " << max_gpu_ms       << " ms\n";
        std::cout << "  Kernel wall    : " << t_wall_ms        << " ms  (barrier-to-barrier)\n";
        std::cout << "  Gather (disp)  : " << t_gather * 1e3 << " ms\n";
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
        std::cout << "====================================================\n\n";

        if (!opt.csv_path.empty()) {
            std::ifstream fin(opt.csv_path);
            bool new_file = !fin.good();
            fin.close();
            std::ofstream csv(opt.csv_path, std::ios::app);
            if (new_file)
                csv << "nranks,height,width,max_disp,radius,kernel_ms,bcast_ms,"
                       "gather_ms,comm_ms,wall_ms\n";
            csv << nranks << "," << H << "," << W << ","
                << p.max_disp << "," << p.radius << ","
                << max_gpu_ms << ","
                << t_bcast  * 1e3 << ","
                << t_gather * 1e3 << ","
                << t_comm_total << ","
                << t_wall_ms << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
