"""PDF reports for Sentinel-2 zone comparisons."""

from __future__ import annotations

from datetime import date
from html import escape
from io import BytesIO
from typing import Any, Mapping

import numpy as np
from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _paragraph(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def _scene_value(metadata: Mapping[str, Any] | None, key: str, default: str = "Unavailable") -> str:
    if metadata is None:
        return default
    value = metadata.get(key)
    if value is None:
        return default
    if key == "cloud_cover":
        return f"{float(value):.1f}%"
    return str(value)


def _preview_flowable(
    image: np.ndarray | None,
    label: str,
    body_style: ParagraphStyle,
    max_width: float,
    max_height: float,
    image_streams: list[BytesIO],
) -> list[object]:
    content: list[object] = [_paragraph(label, body_style), Spacer(1, 6)]
    if image is None:
        content.append(_paragraph("Preview unavailable", body_style))
        return content

    image_stream = BytesIO()
    PillowImage.fromarray(image).save(image_stream, format="PNG")
    image_stream.seek(0)
    image_streams.append(image_stream)
    report_image = Image(image_stream)
    scale = min(max_width / report_image.imageWidth, max_height / report_image.imageHeight)
    report_image.drawWidth = report_image.imageWidth * scale
    report_image.drawHeight = report_image.imageHeight * scale
    content.append(report_image)
    return content


def create_zone_report_pdf(
    *,
    corporation_name: str,
    city_name: str,
    area_name: str,
    latitude: float,
    longitude: float,
    t1_target: date,
    t1_image: np.ndarray | None,
    t1_metadata: Mapping[str, Any] | None,
    t2_target: date,
    t2_image: np.ndarray | None,
    t2_metadata: Mapping[str, Any] | None,
    detection_summary: Mapping[str, int | float] | None = None,
    change_summary: Mapping[str, int | float] | None = None,
    candidate_patches: list[Mapping[str, Any]] | None = None,
    anomaly_image_pairs: list[Mapping[str, Any]] | None = None,
    analysis_note: str | None = None,
) -> bytes:
    """Build a PDF with scene evidence and a model-backed or unavailable summary."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=30,
        leftMargin=30,
        topMargin=28,
        bottomMargin=28,
        title=f"Sentinel-2 Zone Report - {area_name}",
        author="Construction Watch",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ZoneReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#183b2d"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "ZoneReportHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#183b2d"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = styles["BodyText"]
    muted_style = ParagraphStyle("ZoneReportMuted", parent=body_style, textColor=colors.HexColor("#53645b"))

    story: list[object] = [
        _paragraph("Sentinel-2 Zone Monitoring Report", title_style),
        _paragraph(f"{corporation_name} / {city_name} / {area_name}", body_style),
        _paragraph(f"Coordinates: {latitude:.5f}, {longitude:.5f}", body_style),
        _paragraph(f"Generated: {date.today().isoformat()}", muted_style),
        Spacer(1, 10),
        _paragraph("EPOCH T0 / EPOCH T1 imagery", heading_style),
    ]

    scene_rows = [
        ["Scene field", "EPOCH T0 - Past", "EPOCH T1 - Recent"],
        ["Product", _scene_value(t1_metadata, "processing_level"), _scene_value(t2_metadata, "processing_level")],
        ["Target date", t1_target.isoformat(), t2_target.isoformat()],
        ["Acquired", _scene_value(t1_metadata, "acquired"), _scene_value(t2_metadata, "acquired")],
        ["Cloud cover", _scene_value(t1_metadata, "cloud_cover"), _scene_value(t2_metadata, "cloud_cover")],
        ["Scene ID", _scene_value(t1_metadata, "scene_id"), _scene_value(t2_metadata, "scene_id")],
    ]
    scene_table = Table(scene_rows, colWidths=[document.width * 0.22, document.width * 0.39, document.width * 0.39])
    scene_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#183b2d")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5e0d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    image_streams: list[BytesIO] = []
    preview_width = document.width / 2 - 14
    preview_height = 270
    preview_table = Table(
        [
            [
                _preview_flowable(
                    t1_image,
                    f"EPOCH T0 (Sentinel-2 MSI, 1 km radius context) · {_scene_value(t1_metadata, 'acquired')} · cloud {_scene_value(t1_metadata, 'cloud_cover')}",
                    body_style,
                    preview_width,
                    preview_height,
                    image_streams,
                ),
                _preview_flowable(
                    t2_image,
                    f"EPOCH T1 (Sentinel-2 MSI, 1 km radius context) · {_scene_value(t2_metadata, 'acquired')} · cloud {_scene_value(t2_metadata, 'cloud_cover')}",
                    body_style,
                    preview_width,
                    preview_height,
                    image_streams,
                ),
            ]
        ],
        colWidths=[document.width / 2, document.width / 2],
    )
    preview_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5e0d8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5e0d8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    anomaly_total_style = ParagraphStyle(
        "AnomalyTotal",
        parent=heading_style,
        fontSize=18,
        leading=22,
        spaceBefore=8,
        spaceAfter=0,
    )
    story.extend(
        [
            preview_table,
            Spacer(1, 10),
            _paragraph(f"Anomalies identified: {len(candidate_patches or [])}", anomaly_total_style),
            _paragraph(
                "Optical imagery embedded in this report is Sentinel-2 L2A. Candidate footprints are marked on the Epoch T0/T1 crops.",
                muted_style,
            ),
            PageBreak(),
            _paragraph("Analysis details", title_style),
            _paragraph("OPTICAL VERIFICATION", heading_style),
            _paragraph(
                "Embedded Sentinel-2 MSI optical imagery is shown in the Epoch T0/T1 overview and the marked per-anomaly crops below.",
                body_style,
            ),
            _paragraph("EPOCH T0 / EPOCH T1 scene details", heading_style),
            scene_table,
            _paragraph("Spectral change screening", heading_style),
        ]
    )

    if change_summary is None:
        story.append(_paragraph("No paired clear-land spectral comparison was available for this report.", body_style))
    else:
        change_rows = [
            ["Indicator", "Approximate area"],
            ["Clear land compared", f"{float(change_summary.get('valid_area_hectares', 0.0)):.2f} ha"],
            ["Land-clearing candidates", f"{float(change_summary.get('land_clearing_area_hectares', 0.0)):.2f} ha"],
            ["Built-surface gain candidates", f"{float(change_summary.get('built_surface_gain_area_hectares', 0.0)):.2f} ha"],
            ["Mean NDVI change (T2 - T1)", f"{float(change_summary.get('mean_ndvi_delta', 0.0)):+.3f}"],
            ["Mean NDBI change (T2 - T1)", f"{float(change_summary.get('mean_ndbi_delta', 0.0)):+.3f}"],
        ]
        if "sar_relative_orbit" in change_summary:
            change_rows.extend(
                [
                    ["SAR-backed vertical-structure proxy", f"{float(change_summary.get('sar_structure_area_hectares', 0.0)):.2f} ha"],
                    ["Sentinel-1 IW descending relative orbit", f"#{int(change_summary['sar_relative_orbit'])}"],
                    ["Sentinel-1 acquisition dates", f"{change_summary.get('sar_t1_acquired', 'Unavailable')} / {change_summary.get('sar_t2_acquired', 'Unavailable')}"],
                    ["Mean VV backscatter change", f"{float(change_summary.get('sar_mean_vv_change_db', 0.0)):+.2f} dB"],
                    ["VV gain threshold", f">= {float(change_summary.get('sar_vv_gain_threshold_db', 4.0)):.1f} dB"],
                ]
            )
        change_table = Table(change_rows, colWidths=[document.width * 0.68, document.width * 0.32])
        change_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0eb")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5e0d8")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.extend([change_table, Spacer(1, 6)])
    story.append(
        _paragraph(
            "Built-surface gain is a spectral screening signal: NDBI rose by at least "
            f"{float((change_summary or {}).get('built_surface_ndbi_threshold', 0.15)):.2f} while NDVI fell by at least "
            f"{abs(float((change_summary or {}).get('built_surface_ndvi_threshold', -0.10))):.2f} from EPOCH T0 to EPOCH T1. "
            "This can indicate more exposed built or impervious surface; it does not confirm a new building.",
            body_style,
        )
    )

    if candidate_patches:
        anomaly_rows = [["Anomaly", "Signal", "Latitude", "Longitude", "Area (ha)"]]
        anomaly_rows.extend(
            [
                str(patch["patch_id"]),
                str(patch["signal"]),
                f"{float(patch['latitude']):.6f}",
                f"{float(patch['longitude']):.6f}",
                f"{float(patch['area_hectares']):.3f}",
            ]
            for patch in candidate_patches
        )
        anomaly_table = LongTable(
            anomaly_rows,
            colWidths=[document.width * fraction for fraction in (0.11, 0.31, 0.18, 0.18, 0.24)],
            repeatRows=1,
        )
        anomaly_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0eb")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d5e0d8")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("LEADING", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.extend(
            [
                _paragraph(f"Candidate anomalies (all {len(candidate_patches)} connected patches)", heading_style),
                anomaly_table,
                Spacer(1, 6),
            ]
        )
    else:
        story.extend(
            [
                _paragraph("Candidate anomalies", heading_style),
                _paragraph("No connected spectral-change patches were identified.", body_style),
            ]
        )
    if anomaly_image_pairs:
        story.append(_paragraph("Per-anomaly image review", heading_style))
        anomaly_image_streams: list[BytesIO] = []
        anomaly_image_width = document.width / 2 - 12
        for anomaly in anomaly_image_pairs:
            patch = anomaly["patch"]
            story.append(
                _paragraph(
                    f"{patch['patch_id']} · {patch['signal']} · "
                    f"{float(patch['area_hectares']):.3f} ha · "
                    f"{float(patch['latitude']):.6f}, {float(patch['longitude']):.6f}",
                    body_style,
                )
            )
            comparison = Table(
                [[
                    _preview_flowable(
                        anomaly.get("t1_image"), "EPOCH T0 (Sentinel-2 MSI)", body_style,
                        anomaly_image_width, 205, anomaly_image_streams,
                    ),
                    _preview_flowable(
                        anomaly.get("t2_image"), "EPOCH T1 (Sentinel-2 MSI)", body_style,
                        anomaly_image_width, 205, anomaly_image_streams,
                    ),
                ]],
                colWidths=[document.width / 2, document.width / 2],
            )
            comparison.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5e0d8")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5e0d8")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend(
                [
                    comparison,
                    _paragraph(
                        "The shaded red area and yellow outline mark the candidate. Crops include about 200 m of surrounding context where the image bounds allow. "
                        "Sentinel-2 samples at 10 m; enlargement improves viewing size but cannot reveal building height or roof details.",
                        muted_style,
                    ),
                    _paragraph(
                        "Permit verification required: check this coordinate against approved municipal "
                        "building permissions and sanctioned plans before treating the change as unauthorized.",
                        muted_style,
                    ),
                    Spacer(1, 10),
                ]
            )
    story.append(_paragraph("Construction assessment", heading_style))

    if detection_summary is None:
        note = analysis_note or (
            "Permit status has not been checked. Spectral candidates are preliminary screening signals only; "
            "verify every anomaly against approved municipal building permissions and sanctioned plans."
        )
        story.append(_paragraph(note, body_style))
    else:
        detected_pixels = int(detection_summary.get("detected_pixels", 0))
        valid_pixels = int(detection_summary.get("valid_pixels", 0))
        detected_share = float(detection_summary.get("detected_share_percent", 0.0))
        summary_rows = [
            ["Model output", "Result"],
            ["Model-flagged construction candidates", "Present" if detected_pixels else "None flagged"],
            ["Share of analyzed area flagged", f"{detected_share:.2f}%"],
        ]
        summary_table = Table(summary_rows, colWidths=[document.width * 0.68, document.width * 0.32])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0eb")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5e0d8")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.extend(
            [
                summary_table,
                Spacer(1, 6),
                _paragraph(
                    "These are model-flagged spectral change candidates, not a legal determination. "
                    "Verify against permits and authoritative records before enforcement.",
                    muted_style,
                ),
            ]
        )

    document.build(story)
    return output.getvalue()
