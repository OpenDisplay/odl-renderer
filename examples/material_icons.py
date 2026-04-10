"""Material Design Icons example."""

import asyncio

from drawcustom import generate_image


async def main():
    """Generate an image with Material Design Icons."""

    image = await generate_image(
        width=296,
        height=128,
        elements=[
            # Home icon
            {
                "type": "icon",
                "value": "home",
                "x": 50,
                "y": 64,
                "size": 48,
                "color": "black",
                "anchor": "mm",
            },
            # Heart icon
            {
                "type": "icon",
                "value": "heart",
                "x": 150,
                "y": 64,
                "size": 48,
                "color": "red",
                "anchor": "mm",
            },
            # Star icon
            {
                "type": "icon",
                "value": "star",
                "x": 246,
                "y": 64,
                "size": 48,
                "color": "yellow",
                "anchor": "mm",
            },
        ],
        background="white",
    )

    image.save("icons_example.png")
    print("Icons saved to icons_example.png")


if __name__ == "__main__":
    asyncio.run(main())
