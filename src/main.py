import argparse
from src.analyzer import SwingAnalyzer
from src.notifier import LineNotifier

def main():
    parser = argparse.ArgumentParser(description='J-Swing Insight Batch Execution')
    parser.add_argument('--notify-line', action='store_true', help='Send results to LINE Notify')
    args = parser.parse_args()

    print("Starting analysis...")
    analyzer = SwingAnalyzer()
    signals = analyzer.analyze()
    
    print(f"Found {len(signals)} signals.")
    
    if args.notify_line:
        print("Sending LINE notification...")
        notifier = LineNotifier()
        notifier.notify_signals(signals)
    else:
        print("Dry run complete (no LINE notification sent).")
        for s in signals:
            print(f"- {s.ticker}: Close={s.close_price:.1f}, RSI(2)={s.rsi2:.1f}")

if __name__ == "__main__":
    main()
