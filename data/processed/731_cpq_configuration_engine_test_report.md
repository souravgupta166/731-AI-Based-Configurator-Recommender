# 731 CPQ Configuration Engine Test Report

## Source

- Workbook: `data/raw/x731_allgemein_Rev24.xlsx`
- Rows: **1,056**
- Columns: **60**

## Technical fields

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

## Package contexts

- `F731PW`
- `F731TE`
- `F731WD_DualChannel`
- `F731WD_SingleChannel`
- `G731CA`
- `G731ST-HT`
- `G731ST-LT`
- `G731VG`
- `x731`

## Deterministic validation

Technical validity is determined from the 731 source rule rows. ML is not used for technical validation.

## Test result

**PASS**

All basic deterministic engine tests passed.

## Model-code status

**NOT VERIFIED**

The engine does not generate Emerson commercial/model codes. Canonical CFG731 configuration identity remains handled by the existing canonical configuration module.