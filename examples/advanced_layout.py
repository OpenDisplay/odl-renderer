"""Advanced layout with percentage-based positioning."""

import asyncio

from drawcustom import generate_image


async def main():
    """Generate a complex layout using percentages and anchors."""
    image = await generate_image(
        width=400,
        height=300,
        elements=[
            # Header section with background
            {
                "type": "rectangle",
                "x_start": 0,
                "y_start": 0,
                "x_end": "100%",
                "y_end": 60,
                "fill": "#333333",
            },
            # Header text
            {
                "type": "text",
                "value": "Dashboard",
                "x": "50%",
                "y": 30,
                "font": "ppb",
                "size": 32,
                "color": "white",
                "anchor": "mm",
            },
            # Three columns with progress bars
            {
                "type": "text",
                "value": "CPU",
                "x": "16.6%",
                "y": 100,
                "font": "ppb",
                "size": 16,
                "anchor": "mm",
            },
            {
                "type": "progress_bar",
                "x_start": "8.3%",
                "y_start": 120,
                "x_end": "25%",
                "y_end": 145,
                "progress": 75,
                "fill": "accent",
                "background": "white",
                "outline": "black",
                "width": 2,
                "show_percentage": True,
            },
            {
                "type": "text",
                "value": "Memory",
                "x": "50%",
                "y": 100,
                "font": "ppb",
                "size": 16,
                "anchor": "mm",
            },
            {
                "type": "progress_bar",
                "x_start": "41.6%",
                "y_start": 120,
                "x_end": "58.3%",
                "y_end": 145,
                "progress": 25,
                "fill": "accent",
                "background": "white",
                "outline": "black",
                "width": 2,
                "show_percentage": True,
            },
            {
                "type": "text",
                "value": "Disk",
                "x": "83.3%",
                "y": 100,
                "font": "ppb",
                "size": 16,
                "anchor": "mm",
            },
            {
                "type": "progress_bar",
                "x_start": "75%",
                "y_start": 120,
                "x_end": "91.6%",
                "y_end": 145,
                "progress": 75,
                "fill": "accent",
                "background": "white",
                "outline": "black",
                "width": 2,
                "show_percentage": True,
            },
            # Footer with timestamp
            {
                "type": "text",
                "value": "Last updated: 2025-12-28 10:30 AM",
                "x": "50%",
                "y": 280,
                "font": "ppb",
                "size": 12,
                "color": "#666666",
                "anchor": "mm",
            },
        ],
        background="white",
        accent_color="red",
    )

    image.save("advanced_layout.png")
    print("Advanced layout saved to advanced_layout.png")


if __name__ == "__main__":
    asyncio.run(main())
