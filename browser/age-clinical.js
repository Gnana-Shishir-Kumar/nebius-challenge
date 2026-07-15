// Age-adjusted clinical hints, prevalence charts, and heuristic age estimates.

export const PREVALENCE = {
  "Endometrioma / Chocolate Cyst": {
    "<20": 5,
    "20-29": 25,
    "30-39": 40,
    "40-49": 22,
    "50+": 8,
  },
  "Simple Cyst": {
    "<20": 15,
    "20-29": 20,
    "30-39": 20,
    "40-49": 25,
    "50+": 20,
  },
  "Dermoid / Teratoma": {
    "<20": 20,
    "20-29": 35,
    "30-39": 25,
    "40-49": 15,
    "50+": 5,
  },
  "Mucinous Cystadenoma": {
    "<20": 5,
    "20-29": 15,
    "30-39": 25,
    "40-49": 30,
    "50+": 25,
  },
  "Serous Cystadenoma": {
    "<20": 5,
    "20-29": 15,
    "30-39": 25,
    "40-49": 30,
    "50+": 25,
  },
  "Complex / Irregular / Indeterminate": {
    "<20": 5,
    "20-29": 15,
    "30-39": 25,
    "40-49": 30,
    "50+": 25,
  },
};

export const AGE_BANDS = ["<20", "20-29", "30-39", "40-49", "50+"];

export function shapeToLesionType(shapeName) {
  switch (shapeName) {
    case "Round / Unilocular":
      return "Endometrioma / Chocolate Cyst";
    case "Oval / Regular":
      return "Simple Cyst";
    case "Irregular / Lobulated":
      return "Dermoid / Teratoma";
    case "Multilocular / Complex":
      return "Mucinous Cystadenoma";
    default:
      return "Complex / Irregular / Indeterminate";
  }
}

export function ageBandForAge(age) {
  if (age == null || Number.isNaN(age)) return null;
  if (age < 20) return "<20";
  if (age <= 29) return "20-29";
  if (age <= 39) return "30-39";
  if (age <= 49) return "40-49";
  return "50+";
}

/**
 * Age-adjusted clinical banner.
 * Returns { text, risk } or null if age unset / outside message bands.
 * `risk` drives banner color (matches clinical urgency of the age message).
 */
export function ageAdjustedFlag(age) {
  if (age == null || Number.isNaN(age)) return null;
  if (age >= 13 && age <= 24) {
    return {
      text: "Adolescent — functional cyst more likely than endometrioma at this age. Monitor before intervention.",
      risk: "low-moderate",
    };
  }
  if (age >= 25 && age <= 40) {
    return {
      text: "Peak endometrioma prevalence age range — elevated suspicion for endometrioma if unilocular.",
      risk: "moderate",
    };
  }
  if (age >= 41 && age <= 50) {
    return {
      text: "Perimenopausal — consider hemorrhagic cyst alongside endometrioma. CA-125 correlation recommended.",
      risk: "moderate",
    };
  }
  if (age >= 51) {
    return {
      text: "Post-menopausal — any complex cyst warrants urgent specialist review regardless of morphology.",
      risk: "higher",
    };
  }
  return null;
}

/** diameter_cm from area_px at 0.2 mm/px spacing. */
export function diameterCmFromArea(areaPx) {
  if (!areaPx || areaPx <= 0) return 0;
  return 2 * Math.sqrt(areaPx / Math.PI) * 0.02;
}

export function estimateAgeRange(shapeName, areaPx) {
  const d = diameterCmFromArea(areaPx);
  const name = shapeName || "";

  if (name.includes("Unilocular")) {
    return d > 3 ? "25–45" : "20–40";
  }
  if (name.includes("Irregular") || name.includes("Lobulated")) {
    return "18–38";
  }
  if (name.includes("Multilocular") || name.includes("Complex")) {
    return d > 5 ? "35–55" : "30–50";
  }
  return "25–45";
}

export function parseOptionalAge(raw) {
  if (raw === "" || raw == null) return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  if (n < 10 || n > 80) return null;
  return Math.round(n);
}
