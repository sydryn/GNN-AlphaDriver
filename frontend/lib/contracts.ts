export const ASSAYS = ["RNA_SEQ", "ATAC", "CAGE", "SPLICE_SITES"] as const;
export const TISSUES = {
  lung: "Lung",
  liver: "Liver",
  brain: "Brain",
  colon: "Colon · transverse",
} as const;

export type Assay = (typeof ASSAYS)[number];
export type Tissue = keyof typeof TISSUES;

export interface VariantQuery {
  chr: string;
  pos: number;
  ref: string;
  alt: string;
  tissue: Tissue;
  window: "100kb" | "1mb";
  max_points: number;
}

export const EXAMPLE: VariantQuery = {
  chr: "chr22",
  pos: 36201698,
  ref: "A",
  alt: "C",
  tissue: "lung",
  window: "100kb",
  max_points: 512,
};

export const ASSAY_LABEL: Record<Assay, string> = {
  RNA_SEQ: "RNA-seq",
  ATAC: "ATAC-seq",
  CAGE: "CAGE",
  SPLICE_SITES: "Splice sites",
};

export interface Track {
  assay: Assay;
  columns: { name: string; strand: string }[];
  native_resolution_bp: number;
  units: string;
  tissue_specific: boolean;
  bin_edges: number[];
  reference: number[][];
  alternate: number[][];
  delta: number[][];
}

export interface Attribution {
  assay: Assay;
  mean_abs_delta: number;
  max_abs_delta: number;
  masked_logit_change: number;
}

export interface Prediction {
  assembly: "GRCh38";
  variant: VariantQuery;
  interval: { chromosome: string; start: number; end: number };
  tissue_ontology: string;
  signal_source: "mock" | "alphagenome";
  reference_validated: boolean;
  tracks: Track[];
  missing_assays: Assay[];
  score: {
    value: number;
    mode: "demo" | "research";
    calibrated: boolean;
    model_version: string;
    network_version: string;
  };
  attributions: Attribution[];
}

export const integer = new Intl.NumberFormat("en-US");
