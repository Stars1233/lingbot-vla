import os
import yaml
import argparse


def rewrite_robot_config(robot_config, norm_path):
    with open(robot_config, 'r') as f:
        robot_config_data = yaml.safe_load(f)
    f.close()
    robot_config_data['norm_stats'] = norm_path

    with open(robot_config, 'w') as f:
        yaml.dump(robot_config_data, f, sort_keys=False)
    f.close()


def rewrite_norm_config(norm_compute_config_file, robot_name, norm_path, data_path):
    with open(norm_compute_config_file, 'r') as f:
        norm_compute_config = yaml.safe_load(f)
    f.close()
    norm_compute_config['data']['data_name'] = robot_name
    norm_compute_config['data']['train_path'] = data_path
    norm_compute_config['data']['norm_path'] = norm_path

    with open(norm_compute_config_file, 'w') as f:
        yaml.dump(norm_compute_config, f, sort_keys=False)
    f.close()

def rewrite_training_config(training_config_file, robot_name, data_path, output_dir):
    with open(training_config_file, 'r') as f:
        training_config = yaml.safe_load(f)
    f.close()
    training_config['data']['data_name'] = robot_name
    training_config['data']['train_path'] = data_path
    training_config['train']['output_dir'] = output_dir
    

    with open(training_config_file, 'w') as f:
        yaml.dump(training_config, f, sort_keys=False)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="posttraining yaml")
        # Policy configuration
    parser.add_argument(
        "--robot_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--norm_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--norm_compute_yaml",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_yaml",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    
    args = parser.parse_args()

    robot_config = f"configs/robot_configs/{args.robot_name}.yaml"
    rewrite_robot_config(robot_config, os.path.join(args.output_path, 'norm.json'))
    rewrite_norm_config(args.norm_compute_yaml, args.robot_name, args.norm_path, args.data_path)
    rewrite_training_config(args.train_yaml, args.robot_name, args.data_path, args.output_path)

    os.makedirs(args.output_path, exist_ok=True)