"""QR code generation example."""

import asyncio

from drawcustom import generate_image


async def main():
    """Generate an image with a QR code."""
    image = await generate_image(
        width=200,
        height=200,
        elements=[
            {
                "type": "qrcode",
                "data": "https://github.com/g4bri3lDev/drawcustom",
                "x": "50%",
                "y": "50%",
                "size": 150,
                "anchor": "mm",
            },
        ],
        background="white",
    )

    image.save("qrcode_example.png")
    print("QR code saved to qrcode_example.png")


if __name__ == "__main__":
    asyncio.run(main())
