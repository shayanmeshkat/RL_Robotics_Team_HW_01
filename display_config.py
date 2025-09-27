import os
import matplotlib
import warnings
import argparse

# Global visualization flags
SHOW_PLOTS = False
SAVE_PLOTS = True
INTERACTIVE_MODE = False

def configure_display(show_plots=False, save_plots=True, interactive=False):
    """Configure matplotlib and suppress Qt warnings for headless environments"""
    
    global SHOW_PLOTS, SAVE_PLOTS, INTERACTIVE_MODE
    SHOW_PLOTS = show_plots
    SAVE_PLOTS = save_plots
    INTERACTIVE_MODE = interactive
    
    if interactive and show_plots:
        # Use interactive backend for showing plots
        try:
            matplotlib.use('TkAgg')  # Try TkAgg first
        except ImportError:
            try:
                matplotlib.use('Qt5Agg')  # Fallback to Qt5Agg
            except ImportError:
                matplotlib.use('Agg')  # Final fallback to non-interactive
                print("Warning: No interactive backend available, using non-interactive mode")
                SHOW_PLOTS = False
                INTERACTIVE_MODE = False
    else:
        # Set matplotlib to use non-interactive backend
        matplotlib.use('Agg')
    
    # Suppress Qt platform plugin warnings
    if not interactive:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # Suppress specific matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # Set additional environment variables to avoid Qt issues
    os.environ.setdefault('DISPLAY', ':0')
    
    print(f"Display configuration: show_plots={SHOW_PLOTS}, save_plots={SAVE_PLOTS}, interactive={INTERACTIVE_MODE}")

def get_visualization_flags():
    """Return current visualization flags"""
    return SHOW_PLOTS, SAVE_PLOTS, INTERACTIVE_MODE

def parse_args():
    """Parse command line arguments for visualization settings"""
    parser = argparse.ArgumentParser(description='Q-Learning with visualization options')
    parser.add_argument('--show-plots', action='store_true', 
                       help='Show plots in interactive windows')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots to files')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable fully interactive mode')
    
    args = parser.parse_args()
    
    show_plots = args.show_plots or args.interactive
    save_plots = not args.no_save
    interactive = args.interactive
    
    return show_plots, save_plots, interactive

# Call this function at the start of your main scripts
if __name__ == "__main__":
    show, save, interactive = parse_args()
    configure_display(show, save, interactive)
