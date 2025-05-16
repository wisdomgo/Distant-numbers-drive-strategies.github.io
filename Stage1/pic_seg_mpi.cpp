#include <mpi.h> // 引入MPI库，用于并行计算
#include <opencv2/opencv.hpp> // 引入OpenCV库，用于图像处理
#include <iostream>
#include <filesystem> // C++17标准库，文件系统相关操作
#include <chrono> // 计时库，用于记录处理时间
#include <vector>
#include <cstring>

namespace fs = std::filesystem; // 使用更简短的命名空间fs

// 定义图像处理函数，用于将图像分割成指定大小并保存
void processImage(const std::string& imagePath, const std::string& outputDir, int rank, int blockSize) {
    // 读取图像
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) { // 检查图像是否读取成功
        std::cerr << "Error: Could not open or find the image at " << imagePath << std::endl;
        return;
    }

    // 计算图像分割区域的宽度和高度
    int blockWidth = image.cols / blockSize;
    int blockHeight = image.rows / blockSize;

    // 将图像分割成多个区块并保存
    for (int i = 0; i < blockHeight; ++i) {
        for (int j = 0; j < blockWidth; ++j) {
            cv::Rect region(j * blockSize, i * blockSize, blockSize, blockSize);
            cv::Mat subImage = image(region);

            std::string outputPath = outputDir + "/"
                + fs::path(imagePath).stem().string()
                + "_part_" + std::to_string(i) + "_" + std::to_string(j) + ".png";

            cv::imwrite(outputPath, subImage); // 保存分割后的图像
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // 初始化MPI环境

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // 获取进程总数
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 获取当前进程的ID（rank）

    std::string inputDir = "48RTQ"; // 输入图像文件夹
    std::string outputDir = "48RTQ_n_32x32"; // 输出图像文件夹

    int blockSize = 32; // 指定分块大小为32x32

    std::vector<std::string> imagePaths;
    if (rank == 0) { // 仅在主进程（rank 0）中执行
        // 遍历输入文件夹，收集所有PNG图像路径
        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (entry.path().extension() == ".png") {
                imagePaths.push_back(entry.path().string());
            }
        }

        // 若输出文件夹不存在则创建
        if (!fs::exists(outputDir)) {
            fs::create_directory(outputDir);
        }
    }

    int numImages = imagePaths.size();
    // 广播图像数量，使所有进程都能得到该信息
    MPI_Bcast(&numImages, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 将图像路径转换为字符数组并广播
    int maxPathLen = 256; // 假定最大路径长度为256字符
    std::vector<char> allPaths(numImages * maxPathLen, 0);

    if (rank == 0) {
        for (int i = 0; i < numImages; ++i) {
            strncpy(&allPaths[i * maxPathLen], imagePaths[i].c_str(), maxPathLen - 1);
        }
    }

    // 广播所有图像路径字符数组到每个进程
    MPI_Bcast(allPaths.data(), numImages * maxPathLen, MPI_CHAR, 0, MPI_COMM_WORLD);

    // 每个进程将字符数组转换回std::string类型的路径列表
    imagePaths.resize(numImages);
    for (int i = 0; i < numImages; ++i) {
        imagePaths[i] = std::string(&allPaths[i * maxPathLen]);
    }

    // 根据进程数量计算每个进程需要处理的图像数量
    int imagesPerProc = numImages / world_size;
    int remainder = numImages % world_size; // 用于均衡分配多余的图像
    int start = rank * imagesPerProc + std::min(rank, remainder);
    int end = start + imagesPerProc + (rank < remainder ? 1 : 0);

    std::vector<double> times(world_size, 0.0); // 用于记录每个进程的处理时间

    // 每个进程处理其分配的图像
    for (int i = start; i < end; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        processImage(imagePaths[i], outputDir, rank, blockSize);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        times[rank] += duration.count(); // 累加该进程的总处理时间
    }

    double total_time;
    // 使用MPI_Reduce将所有进程的处理时间加总，并将结果汇总到主进程
    MPI_Reduce(&times[rank], &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) { // 主进程输出总处理时间和每个进程的处理时间
        std::cout << "Total processing time: " << total_time << " seconds" << std::endl;
        for (int i = 0; i < world_size; ++i) {
            std::cout << "Process " << i << " time: " << times[i] << " seconds" << std::endl;
        }
    }

    MPI_Finalize(); // 结束MPI环境
    return 0;
}