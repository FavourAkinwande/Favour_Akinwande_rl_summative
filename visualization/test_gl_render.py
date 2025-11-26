"""Quick sanity test for the ModernGL renderer.

Usage:
    python -m scripts.test_gl_render
"""

from environment.food_env import FoodRedistributionEnv


def main() -> None:
    env = FoodRedistributionEnv(render_mode="rgb_array")
    env.reset()
    frame = env.render(mode="rgb_array")
    print("Frame type:", type(frame))
    if hasattr(frame, "shape"):
        print("Shape:", frame.shape)
    if hasattr(frame, "dtype"):
        print("dtype:", frame.dtype)
    env.close()


if __name__ == "__main__":
    main()

