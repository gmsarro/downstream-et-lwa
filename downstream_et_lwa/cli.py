"""Typer entry point exposing every pipeline step as a subcommand."""

from __future__ import annotations

import typer

import downstream_et_lwa.budget.compute_baro_terms
import downstream_et_lwa.budget.compute_dqdt
import downstream_et_lwa.budget.dqdt_to_dtheta_dt
import downstream_et_lwa.budget.dtdt_to_dtheta_dt
import downstream_et_lwa.budget.export_qg_binaries
import downstream_et_lwa.budget.heating_lwa_source
import downstream_et_lwa.budget.strips
import downstream_et_lwa.budget.strips_climatology
import downstream_et_lwa.capacity.daily_interpolation
import downstream_et_lwa.capacity.merra2_climatology
import downstream_et_lwa.capacity.monthly_climatology
import downstream_et_lwa.capacity.stationary_a0
import downstream_et_lwa.capacity.time_resolved
import downstream_et_lwa.classification.mpas_rw
import downstream_et_lwa.classification.rw
import downstream_et_lwa.classification.wb
import downstream_et_lwa.composites.build_table_cases
import downstream_et_lwa.composites.build_tracks
import downstream_et_lwa.composites.merra2_source_composites
import downstream_et_lwa.composites.mpas_qgpv_export
import downstream_et_lwa.composites.mpas_rwb_masks
import downstream_et_lwa.composites.mpas_rwp_pipeline
import downstream_et_lwa.composites.run_composites
import downstream_et_lwa.composites.run_mc_envelope
import downstream_et_lwa.composites.run_rw_stratified
import downstream_et_lwa.composites.run_wb_stratified
import downstream_et_lwa.lh_removal.compare_runs
import downstream_et_lwa.lh_removal.composite_figure
import downstream_et_lwa.lh_removal.composite_figure_rw_strat
import downstream_et_lwa.lh_removal.event_catalog
import downstream_et_lwa.lh_removal.fc_quantiles
import downstream_et_lwa.lh_removal.forward_integrate
import downstream_et_lwa.lh_removal.lwa_climatology_jjason
import downstream_et_lwa.lh_removal.run_population
import downstream_et_lwa.preprocessing.extract_era5_1deg
import downstream_et_lwa.preprocessing.extract_mpas_v250
import downstream_et_lwa.preprocessing.mpas_to_era5_grid
import downstream_et_lwa.preprocessing.mpas_topoclean
import downstream_et_lwa.rwp.climatology
import downstream_et_lwa.rwp.envelope
import downstream_et_lwa.rwp.lwa_climatology
import downstream_et_lwa.rwp.lwa_envelope

app = typer.Typer(help="Downstream ET LWA diagnostics pipeline.")

app.command("extract-era5-1deg")(downstream_et_lwa.preprocessing.extract_era5_1deg.main)
app.command("extract-mpas-v250")(downstream_et_lwa.preprocessing.extract_mpas_v250.main)
app.command("mpas-to-era5-grid")(downstream_et_lwa.preprocessing.mpas_to_era5_grid.main)
app.command("mpas-topoclean")(downstream_et_lwa.preprocessing.mpas_topoclean.main)

app.command("build-rwp-envelope")(downstream_et_lwa.rwp.envelope.main)
app.command("build-rwp-climatology")(downstream_et_lwa.rwp.climatology.main)
app.command("build-lwa-envelope")(downstream_et_lwa.rwp.lwa_envelope.main)
app.command("build-lwa-climatology")(downstream_et_lwa.rwp.lwa_climatology.main)

app.command("compute-baro-terms")(downstream_et_lwa.budget.compute_baro_terms.main)
app.command("compute-dqdt")(downstream_et_lwa.budget.compute_dqdt.main)
app.command("dqdt-to-dtheta-dt")(downstream_et_lwa.budget.dqdt_to_dtheta_dt.main)
app.command("dtdt-to-dtheta-dt")(downstream_et_lwa.budget.dtdt_to_dtheta_dt.main)
app.command("heating-lwa-source")(downstream_et_lwa.budget.heating_lwa_source.main)
app.command("export-qg-binaries")(downstream_et_lwa.budget.export_qg_binaries.main)
app.command("build-budget-strips")(downstream_et_lwa.budget.strips.main)
app.command("build-budget-strips-climatology")(
    downstream_et_lwa.budget.strips_climatology.main)

app.command("compute-stationary-a0")(downstream_et_lwa.capacity.stationary_a0.main)
app.command("compute-carrying-capacity")(
    downstream_et_lwa.capacity.monthly_climatology.main)
app.command("compute-carrying-capacity-time-resolved")(
    downstream_et_lwa.capacity.time_resolved.main)
app.command("compute-carrying-capacity-merra2-djf")(
    downstream_et_lwa.capacity.merra2_climatology.main)
app.command("interpolate-capacity-daily")(
    downstream_et_lwa.capacity.daily_interpolation.main)

app.command("build-tracks")(downstream_et_lwa.composites.build_tracks.main)
app.command("run-composites")(downstream_et_lwa.composites.run_composites.main)
app.command("run-rw-stratified")(downstream_et_lwa.composites.run_rw_stratified.main)
app.command("run-wb-stratified")(downstream_et_lwa.composites.run_wb_stratified.main)
app.command("merra2-source-composites")(
    downstream_et_lwa.composites.merra2_source_composites.main)
app.command("run-mc-envelope")(downstream_et_lwa.composites.run_mc_envelope.main)
app.command("mpas-rwp-pipeline")(downstream_et_lwa.composites.mpas_rwp_pipeline.main)
app.command("mpas-qgpv-export")(downstream_et_lwa.composites.mpas_qgpv_export.main)
app.command("mpas-rwb-masks")(downstream_et_lwa.composites.mpas_rwb_masks.main)
app.command("build-table-cases")(downstream_et_lwa.composites.build_table_cases.main)

app.command("classify-rw")(downstream_et_lwa.classification.rw.main)
app.command("classify-wb")(downstream_et_lwa.classification.wb.main)
app.command("classify-mpas-rw")(downstream_et_lwa.classification.mpas_rw.main)

app.command("lh-event-catalog")(downstream_et_lwa.lh_removal.event_catalog.main)
app.command("lh-forward-integrate")(downstream_et_lwa.lh_removal.forward_integrate.main)
app.command("lh-compare-runs")(downstream_et_lwa.lh_removal.compare_runs.main)
app.command("lh-run-population")(downstream_et_lwa.lh_removal.run_population.main)
app.command("lh-composite-figure")(downstream_et_lwa.lh_removal.composite_figure.main)
app.command("lh-composite-figure-rw-strat")(
    downstream_et_lwa.lh_removal.composite_figure_rw_strat.main)
app.command("lh-fc-quantiles")(downstream_et_lwa.lh_removal.fc_quantiles.main)
app.command("lh-lwa-climatology-jjason")(
    downstream_et_lwa.lh_removal.lwa_climatology_jjason.main)


if __name__ == "__main__":
    app()
