"""Compatibility facade for connectivity and bonded-term enumeration."""

from .topology import assign_image_shifts, compute_winding, enumerate_terms, infer_bonds, validate_term_lifts, winding_rank

__all__ = ["assign_image_shifts", "compute_winding", "enumerate_terms", "infer_bonds", "validate_term_lifts", "winding_rank"]
