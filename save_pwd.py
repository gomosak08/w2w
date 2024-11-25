import os

# Get current working directory
current_dir = os.getcwd()

# Save to a text file
output_file = os.path.join(current_dir, "routes.txt")
with open(output_file, "w") as file:
    file.write(current_dir)

print(f"Current directory saved to {output_file}")