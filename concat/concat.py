import csv
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
import time

def score_matrix(file_path='/mnt/Final/dev-measure/concat/sichuan_granules_scores_1.csv'):
    granule_scores = {}
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            granule_id = row[1]
            granule_id = granule_id.split('_')[0]
            filename = row[2]
            score = float(row[3])
            
            # extract row and column indices from the filename
            parts = filename.split('_')
            row_idx = int(parts[-2]) - 1  # row index (starting from 0)
            col_idx = int(parts[-1]) - 1  # column index (starting from 0)
            
            # create score matrix for each granule_id
            if granule_id not in granule_scores:
                granule_scores[granule_id] = np.zeros((32, 32))  # 32x32 score matrix
            
            # fill in the score for the corresponding block
            granule_scores[granule_id][row_idx, col_idx] = score
    
    return granule_scores

if __name__ == '__main__':
    whole_start = time.time()  # overall start time
    granule_scores = score_matrix()

    # The grid size is equal to the number of large images (10976x10976)
    # Each threading block will process a 32x32 chunk of the large image
    # A threading block contain 1024 threads
    # Each thread will process a 343x343 region of the chunk
    kernel_code = """
    __global__ void score_mapping_kernel(float *scores, float *output, int width, int height, int num_blocks_per_row, int num_images) {
        int image_idx = blockIdx.x;  // current image index
        int block_idx = threadIdx.x + threadIdx.y * blockDim.x;  // current thread index in the block

        int row = block_idx / num_blocks_per_row;  // the row index of the chunk
        int col = block_idx % num_blocks_per_row;  // the column index of the chunk

        if (row < num_blocks_per_row && col < num_blocks_per_row) {
            // find the starting position of the score matrix for the current granule
            float score = scores[image_idx * num_blocks_per_row * num_blocks_per_row + row * num_blocks_per_row + col];
            int start_row = row * 343;
            int start_col = col * 343;

            // compute the shift of the current image in the output array
            int image_offset = image_idx * width * height;

            // fill the 343x343 region of each chunk
            for (int i = 0; i < 343; i++) {
                for (int j = 0; j < 343; j++) {
                    int index = image_offset + (start_row + i) * width + (start_col + j);
                    output[index] = score;
                }
            }
            __syncthreads();
        }
    }
    """

    mod = SourceModule(kernel_code) # compile the CUDA kernel
    score_mapping_kernel = mod.get_function("score_mapping_kernel") # get the kernel function

    scores = score_matrix()  # score dictionary for each granule

    granule_ids = list(scores.keys())
    num_granules = len(granule_ids)  # number of granules (large images)
    num_blocks_per_row = 32  # number of chunks inside each large image
    width = 343 * num_blocks_per_row  # row size of the large image
    height = 343 * num_blocks_per_row  # column size of the large image
    n = num_granules  # number of large images
    print(f'Number of granules to process: {n}')
    # allocate memory for the scores array
    scores_array = np.zeros((n, num_blocks_per_row, num_blocks_per_row), dtype=np.float32)
    for i, granule_id in enumerate(granule_ids):
        scores_array[i] = scores[granule_id]

    # allocate memory on the GPU
    scores_gpu = cuda.mem_alloc(scores_array.nbytes)
    output_gpu = cuda.mem_alloc(width * height * n * np.float32().itemsize)

    # copy the scores array to the GPU
    cuda.memcpy_htod(scores_gpu, scores_array)

    # set the block size and grid size
    block_size = 1024  # 1024 threads per block, 2D block
    grid_size = (n, 1, 1)  # n blocks in the grid, 1D grid

    kernel_start = time.time()
    # call the kernel function
    #  __global__ void score_mapping_kernel(float *scores, float *output, int width, int height, int num_blocks_per_row, int num_images)

    score_mapping_kernel(scores_gpu, 
                         output_gpu, 
                         np.int32(width), np.int32(height), 
                         np.int32(num_blocks_per_row), np.int32(n),
                         block=(block_size, 1, 1), 
                         grid=(grid_size[0], 1, 1))  

    # copy the result back to the CPU
    output = np.zeros((n, height, width), dtype=np.float32)
    cuda.memcpy_dtoh(output, output_gpu)
    kernel_end = time.time()

    for i in range(n):
        image_start = time.time()
        fig, ax = plt.subplots(figsize=(8, 8))

        # 只显示图像，不显示坐标轴
        ax.imshow(output[i], cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=1)
        ax.axis('off')  # 关闭坐标轴

        # 保存图片时去掉边框
        plt.savefig(f'/mnt/Final/concat/img_output/{granule_ids[i]}.png', 
                    bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
        image_end = time.time()
        print(f'Processing time for image {granule_ids[i]}: {image_end - image_start:.2f} seconds')

    whole_end = time.time()
    print(f"CUDA Kernel Execution Time: {kernel_end - kernel_start:.2f} seconds")
    print(f"Total Execution Time: {whole_end - whole_start:.2f} seconds")