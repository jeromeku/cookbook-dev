import json
import os
import re

TOTAL_PAYLOAD=20e9
WARMUP=20
TRIALS=20

def parse_log_files(root_dir, output_file):
    # Regular expressions to match the required patterns
    main_pattern = re.compile(r"world_size=(\d+) bytes=(\d+) total_duration=([\d.]+)")
    xfer_pattern = re.compile(r"xfer time \(ms\): ([\d.]+)")

    # List to collect extracted data
    data = []

    # Recursively search through the directory
    for subdir, _, files in os.walk(root_dir):
        for file in sorted(files):
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Temporary storage for the extracted values
            current_world_size = None
            current_bytes = None
            current_total_duration = None
            xfer_times = []

            # Parse the file line by line
            for line in lines:
                # Check for the main line pattern
                main_match = main_pattern.match(line.strip())
                if main_match:
                    # If a new main pattern is found, process the previous data if it exists
                    if current_world_size is not None:
                        # Append all collected data
                        data.append({
                            # "filename": file_path,
                            "world_size": int(current_world_size),
                            "bytes": int(current_bytes),
                            "total_duration": float(current_total_duration),
                            "xfer_times": xfer_times
                        })
                    # Update current main data
                    current_world_size, current_bytes, current_total_duration = main_match.groups()
                    xfer_times = []  # Reset xfer times for the new main entry

                # Check for xfer time pattern
                xfer_match = xfer_pattern.match(line.strip())
                if xfer_match and current_world_size is not None:
                    xfer_times.append(float(xfer_match.group(1)))

            # Process any remaining data in the file after looping
            if current_world_size is not None:
                data.append({
                    # "filename": file_path,
                    "total_payload": int(TOTAL_PAYLOAD),
                    "warmup_steps": int(WARMUP),
                    "trials": int(TRIALS),
                    "world_size": int(current_world_size),
                    "bytes": int(current_bytes),
                    "total_duration": float(current_total_duration),
                    "xfer_times": xfer_times
                })

    # Write the collected data to the output JSON file
    with open(output_file, 'w') as out_file:
        json.dump(data, out_file, indent=4)

if __name__ == "__main__":
    root_dir = "/home/jeromek/cookbook-dev/logs/all-gather-gdb/pool0_datahall_a"
    output_file = "benchmark_data.json"
    parse_log_files(root_dir, output_file)
    print(f"Data extracted and saved to {output_file}")
