# Arguments shared by every script that runs the data pipeline
def add_data_args(parser):
    parser.add_argument("--dataset-root", required=True,
                        help="Path to dataset: the folder containing the A4C/ and PSAX/ folders")
    parser.add_argument("--processed-root", required=True,
                        help="Folder to save the extracted frames (.npy) to")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Reuse frames already extracted to --processed-root")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes")
