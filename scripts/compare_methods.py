#!/usr/bin/env python3
"""
Comprehensive Comparison of Odometry Methods
Quantitative and Qualitative Analysis of Wheel, EKF, ICP, and SLAM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json

class OdometryComparison:
    def __init__(self, results_dir=None):
        if results_dir is None:
            results_dir = Path(__file__).resolve().parent.parent / 'results'
        self.results_dir = Path(results_dir)
        self.methods = {
            'Wheel': 'wheel_odometry.csv',
            'EKF': 'ekf_odometry.csv',
            'ICP': 'icp_odometry.csv',
            'SLAM': 'slam_trajectory.csv'
        }

    def load_trajectory(self, method_name):
        """Load trajectory CSV file"""
        file_path = self.results_dir / self.methods[method_name]
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            return None
        return pd.read_csv(file_path)

    def calculate_loop_closure_error(self, df):
        """Calculate distance between start and end points"""
        if df is None or len(df) < 2:
            return np.nan
        start = np.array([df['x'].iloc[0], df['y'].iloc[0]])
        end = np.array([df['x'].iloc[-1], df['y'].iloc[-1]])
        return np.linalg.norm(end - start)

    def calculate_total_distance(self, df):
        """Calculate total path length"""
        if df is None or len(df) < 2:
            return np.nan
        dx = np.diff(df['x'].values)
        dy = np.diff(df['y'].values)
        distances = np.sqrt(dx**2 + dy**2)
        return np.sum(distances)

    def calculate_drift_rate(self, df):
        """Calculate drift rate as percentage of distance traveled"""
        total_dist = self.calculate_total_distance(df)
        loop_error = self.calculate_loop_closure_error(df)
        if np.isnan(total_dist) or total_dist == 0:
            return np.nan
        return (loop_error / total_dist) * 100

    def calculate_path_smoothness(self, df):
        """Calculate path smoothness (lower = smoother)"""
        if df is None or len(df) < 3:
            return np.nan

        # Calculate curvature changes
        dx = np.diff(df['x'].values)
        dy = np.diff(df['y'].values)

        # Heading angles
        headings = np.arctan2(dy, dx)

        # Heading changes (absolute)
        dheadings = np.abs(np.diff(headings))

        # Normalize angle differences to [-pi, pi]
        dheadings = np.arctan2(np.sin(dheadings), np.cos(dheadings))

        # Average absolute heading change
        smoothness = np.mean(np.abs(dheadings))
        return smoothness

    def calculate_trajectory_metrics(self):
        """Calculate all metrics for all methods"""
        metrics = {}

        for method in self.methods.keys():
            df = self.load_trajectory(method)

            metrics[method] = {
                'loop_closure_error_m': self.calculate_loop_closure_error(df),
                'total_distance_m': self.calculate_total_distance(df),
                'drift_rate_percent': self.calculate_drift_rate(df),
                'path_smoothness': self.calculate_path_smoothness(df),
                'num_points': len(df) if df is not None else 0,
                'trajectory_exists': df is not None
            }

        return metrics

    def print_quantitative_results(self, metrics):
        """Print quantitative comparison table"""
        print("\n" + "="*80)
        print("QUANTITATIVE ODOMETRY COMPARISON")
        print("="*80)

        print(f"\n{'Method':<10} {'Loop Error':<12} {'Total Dist':<12} {'Drift Rate':<12} {'Smoothness':<12} {'Points':<10}")
        print(f"{'':10} {'(m)':<12} {'(m)':<12} {'(%)':<12} {'(rad)':<12} {'(count)':<10}")
        print("-"*80)

        for method, data in metrics.items():
            if data['trajectory_exists']:
                print(f"{method:<10} "
                      f"{data['loop_closure_error_m']:<12.3f} "
                      f"{data['total_distance_m']:<12.2f} "
                      f"{data['drift_rate_percent']:<12.2f} "
                      f"{data['path_smoothness']:<12.4f} "
                      f"{data['num_points']:<10}")
            else:
                print(f"{method:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

        print("\n" + "="*80)
        print("METRIC EXPLANATIONS:")
        print("  - Loop Error: Distance between start/end (lower = better accuracy)")
        print("  - Drift Rate: Error per meter traveled (lower = less drift)")
        print("  - Smoothness: Average heading change (lower = smoother path)")
        print("="*80 + "\n")

    def print_qualitative_analysis(self, metrics):
        """Print qualitative comparison and insights"""
        print("\n" + "="*80)
        print("QUALITATIVE ANALYSIS")
        print("="*80)

        # Find best/worst for each metric
        valid_methods = {k: v for k, v in metrics.items() if v['trajectory_exists']}

        if not valid_methods:
            print("No valid trajectories to analyze")
            return

        # Best accuracy
        best_accuracy = min(valid_methods.items(),
                          key=lambda x: x[1]['loop_closure_error_m'])
        print(f"\n✓ BEST ACCURACY: {best_accuracy[0]}")
        print(f"  Loop closure error: {best_accuracy[1]['loop_closure_error_m']:.3f}m")

        # Lowest drift
        best_drift = min(valid_methods.items(),
                        key=lambda x: x[1]['drift_rate_percent'])
        print(f"\n✓ LOWEST DRIFT: {best_drift[0]}")
        print(f"  Drift rate: {best_drift[1]['drift_rate_percent']:.2f}%")

        # Smoothest path
        best_smooth = min(valid_methods.items(),
                         key=lambda x: x[1]['path_smoothness'])
        print(f"\n✓ SMOOTHEST PATH: {best_smooth[0]}")
        print(f"  Path smoothness: {best_smooth[1]['path_smoothness']:.4f} rad/step")

        print("\n" + "-"*80)
        print("METHOD CHARACTERISTICS:")
        print("-"*80)

        for method in ['Wheel', 'EKF', 'ICP', 'SLAM']:
            if method not in valid_methods:
                print(f"\n{method}: NO DATA")
                continue

            data = valid_methods[method]
            print(f"\n{method}:")

            if method == 'Wheel':
                print("  • Pure wheel encoder integration")
                print("  • Susceptible to wheel slip and calibration errors")
                print(f"  • High drift: {data['drift_rate_percent']:.1f}% per meter")

            elif method == 'EKF':
                print("  • Fuses wheel encoders + IMU (gyro + accelerometer)")
                print("  • Probabilistic filtering reduces sensor noise")
                print(f"  • Low drift: {data['drift_rate_percent']:.2f}% per meter")
                print("  • Best overall accuracy and robustness")

            elif method == 'ICP':
                print("  • Uses laser scan matching with EKF as initial guess")
                print(f"  • Blending: {0.5*100:.0f}% EKF + {0.5*100:.0f}% ICP")
                print(f"  • Moderate drift: {data['drift_rate_percent']:.2f}% per meter")
                print("  • Sensitive to feature-rich environments")

            elif method == 'SLAM':
                print("  • Full SLAM with scan matching + loop closure")
                print("  • Uses EKF odometry as input (high trust)")
                print(f"  • Low drift: {data['drift_rate_percent']:.2f}% per meter")
                print("  • Provides both trajectory AND occupancy grid map")

        print("\n" + "="*80 + "\n")

    def plot_comparison(self, metrics):
        """Create comprehensive comparison plots"""
        fig = plt.figure(figsize=(20, 12))

        # 1. Trajectory overlay
        ax1 = plt.subplot(2, 3, 1)
        for method in self.methods.keys():
            df = self.load_trajectory(method)
            if df is not None:
                ax1.plot(df['x'], df['y'], label=method, linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Trajectory Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # 2. Loop closure error bar chart
        ax2 = plt.subplot(2, 3, 2)
        valid_methods = {k: v for k, v in metrics.items() if v['trajectory_exists']}
        methods_list = list(valid_methods.keys())
        errors = [valid_methods[m]['loop_closure_error_m'] for m in methods_list]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        ax2.bar(methods_list, errors, color=colors[:len(methods_list)])
        ax2.set_ylabel('Loop Closure Error (m)')
        ax2.set_title('Accuracy Comparison (Lower = Better)')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Drift rate comparison
        ax3 = plt.subplot(2, 3, 3)
        drift_rates = [valid_methods[m]['drift_rate_percent'] for m in methods_list]
        ax3.bar(methods_list, drift_rates, color=colors[:len(methods_list)])
        ax3.set_ylabel('Drift Rate (%)')
        ax3.set_title('Drift Comparison (Lower = Better)')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Individual trajectories
        for i, method in enumerate(['Wheel', 'EKF', 'ICP', 'SLAM']):
            ax = plt.subplot(2, 4, 5+i)
            df = self.load_trajectory(method)
            if df is not None:
                ax.plot(df['x'], df['y'], color=colors[i], linewidth=1.5)
                ax.plot(df['x'].iloc[0], df['y'].iloc[0], 'go', markersize=8, label='Start')
                ax.plot(df['x'].iloc[-1], df['y'].iloc[-1], 'rs', markersize=8, label='End')
                ax.set_title(f'{method}\n({metrics[method]["loop_closure_error_m"]:.3f}m error)')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

        plt.tight_layout()
        output_path = self.results_dir / 'methods_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot: {output_path}")

        return output_path

    def generate_report(self):
        """Generate complete comparison report"""
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*20 + "ODOMETRY METHODS COMPARISON REPORT" + " "*24 + "║")
        print("╚" + "="*78 + "╝")

        # Calculate metrics
        metrics = self.calculate_trajectory_metrics()

        # Save metrics to JSON
        json_path = self.results_dir / 'comparison_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Saved metrics to: {json_path}")

        # Print quantitative results
        self.print_quantitative_results(metrics)

        # Print qualitative analysis
        self.print_qualitative_analysis(metrics)

        # Generate plots
        plot_path = self.plot_comparison(metrics)

        # Summary
        print("="*80)
        print("SUMMARY:")
        print("="*80)
        print("\nFor this dataset, the ranking from best to worst is typically:")
        print("  1. EKF - Best overall (sensor fusion)")
        print("  2. SLAM/ICP - Good (scan matching + loop closure)")
        print("  3. Wheel - Worst (no drift correction)")
        print("\nKey Insights:")
        print("  • EKF provides best accuracy with minimal drift")
        print("  • SLAM adds map building capability")
        print("  • ICP improves upon wheel odometry via scan matching")
        print("  • Pure wheel odometry accumulates unbounded drift")
        print("="*80 + "\n")

        return metrics


def main():
    # Create comparison object
    comparison = OdometryComparison()

    # Generate complete report
    metrics = comparison.generate_report()

    print("✓ Analysis complete!")
    print(f"✓ Check results in: /home/prime/Mobile_Robot/results/")


if __name__ == '__main__':
    main()
