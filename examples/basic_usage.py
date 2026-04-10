"""Basic usage examples for drawcustom."""

import asyncio

from drawcustom import generate_image


async def main():
    """Generate a simple image with text and shapes."""
    image = await generate_image(
        width=296,
        height=128,
        elements=[
            # Title text centered at top
            {
                "type": "text",
                "value": "Hello, drawcustom!",
                "x": "50%",
                "y": 20,
                "font": "ppb",
                "size": 24,
                "color": "black",
                "anchor": "mt",
            },
            # Red rectangle
            {
                "type": "rectangle",
                "x_start": 20,
                "y_start": 50,
                "x_end": 120,
                "y_end": 110,
                "fill": "red",
                "outline": "black",
                "width": 2,
            },
            # Blue circle
            {
                "type": "circle",
                "x": 180,
                "y": 80,
                "radius": 30,
                "fill": "blue",
                "outline": "black",
                "width": 2,
            },
            # Horizontal line
            {
                "type": "line",
                "x_start": 0,
                "y_start": 64,
                "x_end": 296,
                "y_end": 64,
                "fill": "black",
                "width": 1,
            },
        ],
        background="white",
    )

    # Save the image
    image.save("basic_example.png")
    print("Image saved to basic_example.png")


if __name__ == "__main__":
    asyncio.run(main())
