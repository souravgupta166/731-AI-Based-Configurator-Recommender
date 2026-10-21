# 731 CPQ Data Audit

## 1. Audit objective

This audit evaluates whether the 731 source data can support a CPQ-equivalent technical configuration workflow without exposing unsupported or invented configuration fields.

The audit deliberately separates customer-facing technical selections from deterministic backend rules, constant product metadata, legacy fields, and fields requiring further investigation.

## 2. Source inventory

- Source workbook: `data/raw/x731_allgemein_Rev24.xlsx`
- Rows: **1,056**
- Columns: **60**
- Current technical UI fields: **19**

## 3. Current technical UI schema

The following fields are currently represented in the technical configuration schema:

- `devCategory`
- `characteristic`
- `housing`
- `powerSupply`
- `numberOfChannels`
- `explosionApproval`
- `protectionArea`
- `certification`
- `dataInterface`
- `stromSchaltbar`
- `stromEingaenge`
- `temperaturEingaenge`
- `binaerDigitalOpenColl_MN`
- `binaerOpenColl_MP`
- `waveInjector`
- `dev_advMeterVerification`
- `dynamicGasMaster`
- `customUserFluid`
- `steamApplication`

## 4. Field classification

| Classification | Count |
|---|---:|
| CPQ-visible technical | 19 |
| Constant product metadata | 37 |
| Legacy / unused | 1 |
| Backend rule fields | 2 |
| Variable fields not currently in UI | 2 |
| Investigate / backend / model-code | 1 |

### 4.1 Variable source fields not currently in the UI

- `OrgUnitAreaOfDocument`
- `packageFixed_dev`

### 4.2 Constant product metadata

- `Unnamed: 0`
- `marke`
- `installationsart`
- `type`
- `intrinsicSafety`
- `binaerDigitalOptoRelay`
- `binaerDigitalReedRelay`
- `binaerDigitalOpenCollExia_MA`
- `binaerOpenColl_BC`
- `binaerOpenColl_SC`
- `binaerOpenColl30v`
- `binaerOpenCollExib`
- `binaerOpenCollNamur`
- `binaerOptoRelay_BO`
- `binaerOptoRelay_SO`
- `binaerReedRelay_BR`
- `binaerReedRelay_SR`
- `frequenz0_5khz_FO`
- `frequenz0_5khz_FC`
- `frequenz0_5khzNamur`
- `frequenz5_15HzExia`
- `spannung0_10v`
- `spannung0_1v`
- `stromAktiv`
- `stromAktiv24`
- `stromPassiv`
- `stromPassivExib`
- `StromPassivExia`
- `binaerEingaenge`
- `spannungsEingaenge`
- `stromEingaenge_passiv`
- `stromEingaengeExia`
- `temperaturEingaengeExi`
- `inclusiveManual`
- `powerCable`
- `carryingCase`
- `measuringTape`

### 4.3 Legacy / unused fields

- `hardwareAP`

### 4.4 Backend rule fields

- `OrgUnitAreaOfDocument`
- `packageFixed_dev`

### 4.5 Fields requiring investigation

- `tagPlate`

## 5. Model-code status

**NOT VERIFIED.**

This audit does not infer the meaning of Emerson model-code positions from configuration IDs, hashes, column names, observed value combinations, or synthetic configuration identifiers.

The internal `CFG731_*` canonical configuration identifier is treated only as an internal deterministic configuration identity. It is not treated as an Emerson commercial/model code.

## 6. Recommended CPQ architecture

```text
REAL 731 CPQ DATA
        ↓
Complete CPQ field catalogue
        ↓
Allowed values + fixed values
        ↓
Deterministic compatibility / constraint engine
        ↓
Valid configurations
        ↓
ML ranking
        ↓
Recommended valid configuration
        ↓
Verified model code
```

## 7. Recommended implementation sequence

1. Freeze the 731 source workbook as the technical source of truth.
2. Build a field-by-field CPQ catalogue mapping customer-facing labels to source columns.
3. Record the allowed values and fixed values for each field.
4. Map each selectable field to deterministic compatibility / constraint logic.
5. Separate customer-visible selectors from backend metadata and constant product fields.
6. Generate only configurations that pass deterministic technical validation.
7. Run ML ranking only over technically valid configurations.
8. Add exact model-code generation only after the authoritative model-code mapping is verified.
9. Update the Streamlit UI from the backend CPQ catalogue rather than hard-coding unsupported values.

## 8. Supporting artifact inventory

| Artifact | Status | Rows | Columns |
|---|---|---:|---:|
| Normalized rule matrix | FOUND | 63,360 | 6 |
| Extracted rules | FOUND | 12,424 | 6 |
| Synthetic configurations | FOUND | 5,000 | 34 |
| Knowledge graph nodes | FOUND | 8,261 | 14 |
| Knowledge graph relationships | FOUND | 83,409 | 7 |

## 9. Generated outputs

- `data/processed/731_cpq_data_audit.csv`
- `data/processed/731_cpq_data_audit_summary.json`
- `data/processed/731_cpq_data_audit_report.md`
- `data/processed/731_cpq_value_profile_top20.csv`

## 10. Audit conclusion

The 731 source data contains substantially more information than the current 19-field technical UI schema. However, additional source columns must not automatically be exposed as customer-facing CPQ selectors.

The next implementation step is therefore a controlled field-by-field CPQ mapping rather than a simple one-to-one exposure of all 60 source columns.

The model-code generation layer remains blocked until an authoritative mapping between configuration choices and the actual Emerson/CPQ model code has been verified.
