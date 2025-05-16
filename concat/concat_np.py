import csv
import numpy as np
import matplotlib.pyplot as plt
import time

def load_scores(file_path='/mnt/Final/dev-measure/concat/sichuan_granules_scores_1.csv'):
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

def stitch_images(granule_scores, block_size=343):
    stitched_images = {}
    num_blocks_per_row = 32
    large_image_size = block_size * num_blocks_per_row  # 10976x10976

    for granule_id, score_matrix in granule_scores.items():
        print(f'Stitching image {granule_id}')
        large_image = np.zeros((large_image_size, large_image_size), dtype=np.float32)

        for row in range(num_blocks_per_row):
            for col in range(num_blocks_per_row):
                score = score_matrix[row, col]
                for i in range(row * block_size, (row + 1) * block_size):
                    for j in range(col * block_size, (col + 1) * block_size):
                        large_image[i, j] = score

        stitched_images[granule_id] = large_image

    return stitched_images

if __name__ == '__main__':
    whole_start = time.time()  # overall start time

    # Load scores from CSV file
    scores = load_scores()

    # Stitch images
    stitch_start = time.time()
    stitched_images = stitch_images(scores)
    stitch_end = time.time()

    # Save images
    for granule_id, large_image in stitched_images.items():
        image_start = time.time()
        fig, ax = plt.subplots(figsize=(8, 8))

        # Display image without axes
        ax.imshow(large_image, cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=1)
        ax.axis('off')  # Turn off axes

        # Save image without borders
        plt.savefig(f'/mnt/Final/concat/img_output/{granule_id}.png', 
                    bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
        image_end = time.time()
        print(f'Processing time for image {granule_id}: {image_end - image_start:.2f} seconds')

    whole_end = time.time()
    print(f"Stitching Execution Time: {stitch_end - stitch_start:.2f} seconds")
    print(f"Total Execution Time: {whole_end - whole_start:.2f} seconds")
