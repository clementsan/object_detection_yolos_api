import requests
import io
import json
import argparse
import sys


# Default examples
# api_url = "http://0.0.0.0:8000/api/v1/detect"
# file = "../tests/data/savanna.jpg"


def arg_parser():
    """Parse arguments"""

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Object detection inference via API call')
    # Add arguments
    parser.add_argument('-u', '--url', type=str, help='URL to the server (with endpoint location)', required=True)
    parser.add_argument('-f', '--file', type=str, help='Path to the input image file', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    return parser


def main(args=None):
    """Main function"""

    args = arg_parser().parse_args(args)
    # Use the arguments
    if args.verbose:
        print(f'Input file: {args.file}')

    # Load image
    with open(args.file, 'rb') as image_file:
        image_data = image_file.read()

    # Send request to API
    response = requests.post(args.url, files={'image': image_data})

    if response.status_code == 200:
        # Process the response
        processed_data = json.loads(response.content)
        print('processed_data', processed_data)
    else:
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
