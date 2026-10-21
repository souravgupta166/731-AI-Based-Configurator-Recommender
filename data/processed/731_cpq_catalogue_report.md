# 731 CPQ Catalogue

## Source

- Workbook: `data/raw/x731_allgemein_Rev24.xlsx`
- Sheet: `x731_allgemein`
- Rows: **1,056**
- Columns: **60**

## 1. Customer-facing technical fields

These fields form the current technical CPQ catalogue.

| Field | Label | Values | Count |
|---|---|---|---:|
| `devCategory` | Device Category | F, G, H, S | 4 |
| `characteristic` | Characteristic | CA, GP, PW, ST, TE, VG, WD | 7 |
| `housing` | Housing | AL, ST | 2 |
| `powerSupply` | Power Supply | 1, 4 | 2 |
| `numberOfChannels` | Number of Channels | 1, 2 | 2 |
| `explosionApproval` | Explosion Approval | A, E, F, J, K, N, P, S | 8 |
| `protectionArea` | Protection Area | 2, N | 2 |
| `certification` | Certification | CN, FN, MN, NN, WN | 5 |
| `dataInterface` | Data Interface | BM, BT, DB, FF, HS, MR, MT, NN, PA | 9 |
| `stromSchaltbar` | Switchable Current | CS1, CS2, CS3, CS4, NOVALUE | 5 |
| `stromEingaenge` | Current Inputs | IS1, IS2, IS3, IS4, NOVALUE | 5 |
| `temperaturEingaenge` | Temperature Inputs | NOVALUE, TT1, TT2, TT3, TT4 | 5 |
| `binaerDigitalOpenColl_MN` | Binary Digital Open Collector MN | MN1, MN2, MN3, MN4, NOVALUE | 5 |
| `binaerOpenColl_MP` | Binary Open Collector MP | MP1, MP2, MP3, MP4, NOVALUE | 5 |
| `waveInjector` | Wave Injector | FWI, FWY, NOVALUE | 3 |
| `dev_advMeterVerification` | Advanced Meter Verification | 1, NOVALUE | 2 |
| `dynamicGasMaster` | Dynamic Gas Master | 1, NOVALUE | 2 |
| `customUserFluid` | Custom User Fluid | 1, NOVALUE | 2 |
| `steamApplication` | Steam Application | H, N, NOVALUE | 3 |

## 2. Backend CPQ context

### `packageFixed_dev`

- Role: **package_context**
- Customer selectable: **No**
- Observed values: F731PW, F731TE, F731WD_DualChannel, F731WD_SingleChannel, G731CA, G731ST-HT, G731ST-LT, G731VG, NOVALUE, x731

### `OrgUnitAreaOfDocument`

- Role: **source_metadata**
- Customer selectable: **No**
- Observed values: 0, 1, 2, 3

## 3. Package contexts

Package contexts are backend configuration contexts. They are not automatically customer-facing selectors.

| Package | Source rows |
|---|---:|
| `F731PW` | 20 |
| `F731TE` | 26 |
| `F731WD_DualChannel` | 12 |
| `F731WD_SingleChannel` | 8 |
| `G731CA` | 24 |
| `G731ST-HT` | 133 |
| `G731ST-LT` | 133 |
| `G731VG` | 4 |
| `x731` | 696 |

## 4. Product/source metadata

- `marke` → ['flexim']
- `installationsart` → ['stationaer']
- `type` → ['731']
- `intrinsicSafety` → ['N']
- `inclusiveManual` → ['QI']
- `powerCable` → ['NOVALUE']
- `carryingCase` → ['NOVALUE']
- `measuringTape` → ['NOVALUE']

## 5. Legacy / unused fields

- `analogInput` → []
- `analogInput2` → []
- `analogOutput` → []
- `analogOutput2` → []
- `digitalInput` → []
- `digitalInput2` → []
- `digitalOutput` → []
- `digitalOutput2` → []
- `hardwareAP` → ['NOVALUE']

## 6. Model-code status

**NOT VERIFIED**

This catalogue deliberately does not infer the meaning of Emerson model-code positions.

The internal CFG731_* canonical configuration identifier must not be treated as an Emerson commercial/model code.

## 7. Configuration architecture

```text
Customer RFQ
    ↓
Structured technical requirements
    ↓
731 CPQ Catalogue
    ↓
Package/context resolution
    ↓
Deterministic constraint validation
    ↓
Valid configurations
    ↓
ML ranking
    ↓
Recommended valid configuration
    ↓
Verified model code
```
