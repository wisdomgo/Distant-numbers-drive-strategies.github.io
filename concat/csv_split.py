import pandas as pd
import sys

def split_csv(input_file, output_file1, output_file2, chunksize=15):
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 检查必要的列是否存在
        required_columns = {'direc', 'y_x', 'score'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"输入文件缺少必要的列: {missing}")
        
        # 获取唯一的granule编号
        unique_direc = df['direc'].unique()
        total_granules = len(unique_direc)
        
        if total_granules != 30:
            print(f"警告: 'direc'列中发现{total_granules}个唯一的granule编号，而预期是30个。")
        
        # 分割granule编号
        first_chunk = unique_direc[:chunksize]
        second_chunk = unique_direc[chunksize:]
        
        # 过滤数据
        df_part1 = df[df['direc'].isin(first_chunk)]
        df_part2 = df[df['direc'].isin(second_chunk)]
        
        # 保存到新的CSV文件
        df_part1.to_csv(output_file1, index=False)
        df_part2.to_csv(output_file2, index=False)
        
        print(f"成功将文件分割为:\n1. {output_file1} (包含{len(first_chunk)}个granules)\n2. {output_file2} (包含{len(second_chunk)}个granules)")
    
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到。请确保文件路径正确。")
    except pd.errors.EmptyDataError:
        print("错误: 输入的CSV文件是空的。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 输入和输出文件名
    input_csv = 'sichuan_granules_scores.csv'
    output_csv1 = 'sichuan_granules_scores_part1.csv'
    output_csv2 = 'sichuan_granules_scores_part2.csv'
    
    # 调用分割函数
    split_csv(input_csv, output_csv1, output_csv2)
