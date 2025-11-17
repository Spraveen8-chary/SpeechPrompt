# main.py - entry point for the PromptSpeech framework
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="PromptSpeech Unified Framework CLI")
    parser.add_argument('--mode', choices=['prepare', 'train', 'eval', 'generate'], default='prepare')
    args = parser.parse_args()

    print(f"🚀 Running PromptSpeech mode: {args.mode}")
    print("➡️ Implement each mode in scripts/ or src/ accordingly.")
    print("Example: scripts/run_training.py or src/training/train_prompt.py")

if __name__ == '__main__':
    main()
