# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os
from pathlib import Path

# Function to rename multiple files
def main():

    for filename in os.listdir("."):
        if filename.endswith("_fg.png"):
            src = filename
            dst = filename.split(".")[0].replace("_fg", "_label.png")
            # print(filename.split(".")[0])
            os.rename(src, dst)


# Driver Code
if __name__ == "__main__":

    # Calling main() function
    main()
