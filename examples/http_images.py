"""HTTP image loading example."""

import asyncio

import aiohttp

from drawcustom import generate_image


async def main():
    """Generate an image with HTTP-loaded content."""

    # Reuse aiohttp session for efficiency
    async with aiohttp.ClientSession() as session:
        image = await generate_image(
            width=400,
            height=300,
            elements=[
                # Load image from HTTP
                {
                    "type": "dlimg",
                    "url": "https://picsum.photos/200/150",  # Random placeholder image
                    "x": "0",
                    "y": "0",
                    "xsize": 200,
                    "ysize": 150,
                },
                # Add text overlay
                {
                    "type": "text",
                    "value": "Image from HTTP",
                    "x": "50%",
                    "y": 280,
                    "font": "ppb",
                    "size": 20,
                    "color": "black",
                    "anchor": "mm",
                },
            ],
            background="white",
            session=session,  # Pass session for connection reuse
        )

    image.save("http_image_example.png")
    print("HTTP image example saved to http_image_example.png")


if __name__ == "__main__":
    asyncio.run(main())
