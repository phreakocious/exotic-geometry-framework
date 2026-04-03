#!/usr/bin/env python3
"""
Download O.IRIS EEG datasets from OpenNeuro S3 (no auth required).

ds001787: Meditation EEG (24 subjects, BDF, 64ch)
ds003768: Sleep/Rest EEG (33 subjects, BrainVision, 32ch)

Downloads EEG files only (skips fMRI/MRI). Stores in data/eeg/oiris/.
"""

import os
import sys
import boto3
from botocore import UNSIGNED
from botocore.config import Config

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'eeg', 'oiris')

N_SUBJECTS_MEDITATION = 10  # of 24
N_SUBJECTS_SLEEP = 10       # of 33

def download_file(s3, bucket, key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        # Check size matches
        resp = s3.head_object(Bucket=bucket, Key=key)
        remote_size = resp['ContentLength']
        local_size = os.path.getsize(local_path)
        if local_size == remote_size:
            return False  # already downloaded
    print(f"  Downloading {key} ...", end=" ", flush=True)
    s3.download_file(bucket, key, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"{size_mb:.1f} MB")
    return True


def download_meditation(s3, n_subjects):
    """Download ds001787 meditation BDF files."""
    print(f"\n{'='*60}")
    print(f"ds001787: MEDITATION EEG ({n_subjects} subjects)")
    print(f"{'='*60}")

    bucket = 'openneuro.org'
    dataset = 'ds001787'
    downloaded = 0

    for subj in range(1, n_subjects + 1):
        subj_id = f"sub-{subj:03d}"
        # Session 1 only (session 2 is repeat)
        for ses in [1]:
            ses_id = f"ses-{ses:02d}"
            prefix = f"{dataset}/{subj_id}/{ses_id}/eeg/"

            # List files in this prefix
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            for obj in resp.get('Contents', []):
                key = obj['Key']
                fname = key.split('/')[-1]
                # Only download .bdf and .json (skip events.tsv for now)
                if fname.endswith(('.bdf', '.json')):
                    local = os.path.join(DATA_DIR, dataset, subj_id,
                                         ses_id, 'eeg', fname)
                    if download_file(s3, bucket, key, local):
                        downloaded += 1

    print(f"  Meditation: {downloaded} new files downloaded")
    return downloaded


def download_sleep(s3, n_subjects):
    """Download ds003768 sleep/rest EEG files (BrainVision format)."""
    print(f"\n{'='*60}")
    print(f"ds003768: SLEEP/REST EEG ({n_subjects} subjects)")
    print(f"{'='*60}")

    bucket = 'openneuro.org'
    dataset = 'ds003768'
    downloaded = 0

    for subj in range(1, n_subjects + 1):
        subj_id = f"sub-{subj:02d}"
        prefix = f"{dataset}/{subj_id}/eeg/"

        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)
        for obj in resp.get('Contents', []):
            key = obj['Key']
            fname = key.split('/')[-1]
            # Download all EEG files (.eeg, .vhdr, .vmrk)
            if fname.endswith(('.eeg', '.vhdr', '.vmrk')):
                local = os.path.join(DATA_DIR, dataset, subj_id, 'eeg', fname)
                if download_file(s3, bucket, key, local):
                    downloaded += 1

    # Also grab task JSON files from root
    for task_file in ['task-rest_eeg.json', 'task-sleep_eeg.json']:
        key = f"{dataset}/{task_file}"
        local = os.path.join(DATA_DIR, dataset, task_file)
        download_file(s3, bucket, key, local)

    print(f"  Sleep/Rest: {downloaded} new files downloaded")
    return downloaded


def main():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    os.makedirs(DATA_DIR, exist_ok=True)

    print("O.IRIS Data Download")
    print(f"Target: {DATA_DIR}")

    n_med = download_meditation(s3, N_SUBJECTS_MEDITATION)
    n_sleep = download_sleep(s3, N_SUBJECTS_SLEEP)

    # Summary
    total_size = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    print(f"  Location: {DATA_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
