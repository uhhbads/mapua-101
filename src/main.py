"""
Main entry point for the Campus Booth AR Camera application.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from camera import Camera
from display import Display
from filters import FilterManager, GPAFilter, DogEarFilter, Y2KFilter, CustomFrameFilter


def main() -> int:
    """Main application loop."""
    print("=" * 50)
    print("    CAMPUS BOOTH AR CAMERA v0.1.0")
    print("=" * 50)
    print()
    print("Controls:")
    print("  SPACE      - Next filter")
    print("  1-4        - Select filter directly")
    print("  A          - Toggle auto-rotate")
    print("  H          - Show/hide instructions")
    print("  F          - Toggle fullscreen")
    print("  Q / ESC    - Quit")
    print()
    print("-" * 50)

    # Load configuration
    config_path = Path(__file__).parent.parent / "config.json"
    config = Config.load(config_path)

    # Initialize camera
    camera = Camera(
        camera_index=config.camera_index,
        width=config.capture_width,
        height=config.capture_height,
        target_fps=config.target_fps,
    )

    if not camera.open():
        print("ERROR: Could not open camera!")
        print(f"Tried camera index: {config.camera_index}")
        return 1

    print(f"[OK] Camera opened: {config.capture_width}x{config.capture_height} @ {config.target_fps} FPS")

    # Initialize display
    display = Display(
        fullscreen=config.display_fullscreen,
        show_fps=config.show_fps,
        show_instructions=True,
    )

    # Initialize filters
    filter_manager = FilterManager()
    filter_manager.add_filter(GPAFilter(
        gpa_refresh_interval=config.gpa_refresh_interval_seconds
    ))
    filter_manager.add_filter(DogEarFilter())
    filter_manager.add_filter(Y2KFilter())
    filter_manager.add_filter(CustomFrameFilter())

    print(f"[OK] Loaded {filter_manager.count} filter(s)")
    print(f"[OK] Active filter: {filter_manager.current_name}")

    # Auto-rotate state
    auto_rotate = config.filter_auto_rotate
    rotate_interval = config.filter_rotate_interval_seconds
    last_rotate_time = time.time()

    if auto_rotate:
        print(f"[OK] Auto-rotate enabled: every {rotate_interval}s")

    print()
    print(">>> Starting live feed... <<<")
    print()

    try:
        while True:
            current_time = time.time()

            # Auto-rotate filters
            if auto_rotate and (current_time - last_rotate_time) >= rotate_interval:
                filter_manager.next_filter()
                last_rotate_time = current_time
                display.reset_instructions()

            # Capture frame
            ret, frame = camera.read()

            if not ret or frame is None:
                continue

            # Apply current filter
            filtered_frame = filter_manager.process(frame)

            # Render frame with overlays
            display.render(
                filtered_frame,
                fps=camera.fps,
                filter_name=filter_manager.current_name,
                filter_index=filter_manager.current_index,
                filter_count=filter_manager.count,
                auto_rotate=auto_rotate,
            )

            # Handle key presses
            key = display.check_key(1)
            
            # Exit keys
            if key == ord("q") or key == ord("Q") or key == 27:  # Q or ESC
                print("\n[EXIT] Quit requested.")
                break
            
            # Filter switching
            elif key == ord(" "):  # Space - next filter
                filter_manager.next_filter()
                last_rotate_time = current_time  # Reset auto-rotate timer
                display.reset_instructions()
                print(f"[FILTER] {filter_manager.current_name}")
            
            elif key == ord("1"):
                if filter_manager.set_filter(0):
                    last_rotate_time = current_time
                    display.reset_instructions()
                    print(f"[FILTER] {filter_manager.current_name}")
            
            elif key == ord("2"):
                if filter_manager.set_filter(1):
                    last_rotate_time = current_time
                    display.reset_instructions()
                    print(f"[FILTER] {filter_manager.current_name}")
            
            elif key == ord("3"):
                if filter_manager.set_filter(2):
                    last_rotate_time = current_time
                    display.reset_instructions()
                    print(f"[FILTER] {filter_manager.current_name}")
            
            elif key == ord("4"):
                if filter_manager.set_filter(3):
                    last_rotate_time = current_time
                    display.reset_instructions()
                    print(f"[FILTER] {filter_manager.current_name}")

            # Toggle auto-rotate
            elif key == ord("a") or key == ord("A"):
                auto_rotate = not auto_rotate
                last_rotate_time = current_time
                status = "ON" if auto_rotate else "OFF"
                print(f"[AUTO-ROTATE] {status}")

            # Toggle instructions
            elif key == ord("h") or key == ord("H"):
                display.show_instructions = not display.show_instructions
                if display.show_instructions:
                    display.reset_instructions()
                status = "ON" if display.show_instructions else "OFF"
                print(f"[INSTRUCTIONS] {status}")

            # Toggle fullscreen
            elif key == ord("f") or key == ord("F"):
                display.fullscreen = not display.fullscreen
                display.destroy()
                display.create_window()
                status = "ON" if display.fullscreen else "OFF"
                print(f"[FULLSCREEN] {status}")

    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user.")

    finally:
        # Cleanup
        print("[CLEANUP] Releasing resources...")
        filter_manager.release_all()
        camera.release()
        display.destroy()

    print("[DONE] Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
