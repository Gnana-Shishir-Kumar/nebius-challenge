// Post-segmentation morphology (IOTA-inspired) — mirrors endpoint/analysis.py.

const EMPTY = {
  shape_name: "Indeterminate",
  risk_level: "low",
  clinical_hint: "No lesion contour found",
  circularity: 0,
  solidity: 0,
  aspect_ratio: 0,
  lobe_count: 0,
  area_px: 0,
};

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function polygonArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    a += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
  }
  return Math.abs(a) / 2;
}

function perimeter(pts) {
  let p = 0;
  for (let i = 0; i < pts.length; i++) {
    p += dist(pts[i], pts[(i + 1) % pts.length]);
  }
  return p;
}

function cross(o, a, b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function convexHull(points) {
  if (points.length < 3) return points.slice();
  const pts = points
    .slice()
    .sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/** Moore neighborhood border follow on binary Uint8Array (1 = foreground). */
function extractLargestContour(binary, width, height) {
  const idx = (x, y) => y * width + x;
  const visited = new Uint8Array(binary.length);
  let best = null;
  let bestArea = 0;

  const dirs = [
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
  ];

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = idx(x, y);
      if (!binary[i] || visited[i]) continue;
      // require an empty neighbor (edge pixel)
      let isEdge = false;
      for (const [dx, dy] of dirs) {
        if (!binary[idx(x + dx, y + dy)]) {
          isEdge = true;
          break;
        }
      }
      if (!isEdge) continue;

      const contour = [];
      let cx = x;
      let cy = y;
      let dir = 0;
      let guard = 0;
      const maxSteps = width * height * 2;

      do {
        contour.push({ x: cx, y: cy });
        visited[idx(cx, cy)] = 1;
        let found = false;
        for (let k = 0; k < 8; k++) {
          const nd = (dir + 6 + k) % 8; // prefer left-turn
          const nx = cx + dirs[nd][0];
          const ny = cy + dirs[nd][1];
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          if (binary[idx(nx, ny)]) {
            cx = nx;
            cy = ny;
            dir = nd;
            found = true;
            break;
          }
        }
        if (!found) break;
        guard += 1;
      } while (!(cx === x && cy === y) && guard < maxSteps);

      if (contour.length < 8) continue;
      const area = polygonArea(contour);
      if (area > bestArea) {
        bestArea = area;
        best = contour;
      }
    }
  }

  return best;
}

function simplifyContour(pts, step = 2) {
  if (pts.length <= 32) return pts;
  const out = [];
  for (let i = 0; i < pts.length; i += step) out.push(pts[i]);
  return out.length >= 8 ? out : pts;
}

function countLobes(contour) {
  if (contour.length < 5) return 1;
  const hull = convexHull(contour);
  if (hull.length < 3) return 1;

  // Approximate convexity defects: deep inward recesses vs hull edges.
  let significant = 0;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    const len = dist(a, b);
    if (len < 1e-3) continue;
    let maxDepth = 0;
    for (const p of contour) {
      // perpendicular distance from point to hull edge
      const area2 = Math.abs(cross(a, b, p));
      const depth = area2 / len;
      // only consider points "inside" relative to hull (signed)
      const signed = cross(a, b, p);
      if (signed < 0 && depth > maxDepth) maxDepth = depth;
    }
    if (maxDepth > 2.0) significant += 1;
  }
  return Math.max(1, significant + 1);
}

function boundingAspect(contour) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of contour) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const w = Math.max(1, maxX - minX);
  const h = Math.max(1, maxY - minY);
  return Math.max(w, h) / Math.min(w, h);
}

function classifyIota(circularity, solidity, aspectRatio, lobeCount) {
  if (lobeCount >= 5 || (solidity < 0.75 && lobeCount >= 3)) {
    return {
      shape_name: "Multilocular / Complex",
      risk_level: "higher",
      clinical_hint:
        "Multiple lobulations or low solidity suggest a multilocular / complex outline " +
        "(IOTA-inspired morphology). Further expert review recommended.",
    };
  }
  if (lobeCount >= 3 || solidity < 0.85 || circularity < 0.55) {
    return {
      shape_name: "Irregular / Lobulated",
      risk_level: "moderate",
      clinical_hint:
        "Irregular or lobulated contour with reduced circularity/solidity " +
        "(IOTA-inspired). Morphology is non-smooth — interpret cautiously.",
    };
  }
  if (circularity >= 0.85 && solidity >= 0.9 && lobeCount <= 2 && aspectRatio <= 1.35) {
    return {
      shape_name: "Round / Unilocular",
      risk_level: "low",
      clinical_hint:
        "Nearly circular, solid contour consistent with a simple unilocular outline " +
        "(IOTA-inspired morphology).",
    };
  }
  return {
    shape_name: "Oval / Regular",
    risk_level: "low-moderate",
    clinical_hint:
      "Elongated or mildly asymmetric but regular contour " +
      "(IOTA-inspired oval / regular morphology).",
  };
}

/**
 * @param {Float32Array|Uint8Array|number[]} maskData probabilities or binary
 * @param {number} width
 * @param {number} height
 * @param {number} threshold
 */
export function analyzeMaskShape(maskData, width, height, threshold = 0.5) {
  const binary = new Uint8Array(width * height);
  let fg = 0;
  for (let i = 0; i < binary.length; i++) {
    const v = maskData[i];
    const on = v > 1 ? v > 127 : v > threshold;
    if (on) {
      binary[i] = 1;
      fg += 1;
    }
  }
  if (fg < 16) {
    return { ...EMPTY, clinical_hint: "Lesion area too small for reliable morphology" };
  }

  let contour = extractLargestContour(binary, width, height);
  if (!contour) {
    return { ...EMPTY };
  }
  contour = simplifyContour(contour, 2);

  const area = polygonArea(contour);
  if (area < 16) {
    return { ...EMPTY, clinical_hint: "Lesion area too small for reliable morphology" };
  }

  const peri = perimeter(contour);
  let circularity = peri > 1e-6 ? (4 * Math.PI * area) / (peri * peri) : 0;
  circularity = Math.min(1.5, Math.max(0, circularity));

  const hull = convexHull(contour);
  const hullArea = polygonArea(hull) || 1;
  const solidity = Math.min(1, Math.max(0, area / hullArea));
  const aspectRatio = boundingAspect(contour);
  const lobeCount = countLobes(contour);
  const labels = classifyIota(circularity, solidity, aspectRatio, lobeCount);

  return {
    ...labels,
    circularity: Math.round(circularity * 1000) / 1000,
    solidity: Math.round(solidity * 1000) / 1000,
    aspect_ratio: Math.round(aspectRatio * 1000) / 1000,
    lobe_count: lobeCount,
    area_px: Math.round(area),
  };
}
