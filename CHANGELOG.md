# Changelog

## [0.5.6](https://github.com/OpenDisplay/odl-renderer/compare/odl-renderer-v0.5.5...odl-renderer-v0.5.6) (2026-04-17)


### Bug Fixes

* **plot:** use local timezone ([17995f0](https://github.com/OpenDisplay/odl-renderer/commit/17995f037c042c817f2fc703d5b1a11408362d86))

## [0.5.5](https://github.com/OpenDisplay/odl-renderer/compare/odl-renderer-v0.5.4...odl-renderer-v0.5.5) (2026-04-12)


### Documentation

* add black border around each image ([15e93dc](https://github.com/OpenDisplay/odl-renderer/commit/15e93dc7901038d49d8d996898a215cdf9200eba))

## [0.5.4](https://github.com/OpenDisplay/odl-renderer/compare/odl-renderer-v0.5.3...odl-renderer-v0.5.4) (2026-04-12)


### Documentation

* update image paths for pypi ([ddd35c5](https://github.com/OpenDisplay/odl-renderer/commit/ddd35c57ecf39ee2014c21829e6aaedc726d2084))
* update plot ([5001231](https://github.com/OpenDisplay/odl-renderer/commit/5001231e34f69adf9a053fefc982d9a5b01c4f33))

## [0.5.3](https://github.com/OpenDisplay/odl-renderer/compare/odl-renderer-v0.5.2...odl-renderer-v0.5.3) (2026-04-12)


### Documentation

* add generateable screenshots ([fecb2c0](https://github.com/OpenDisplay/odl-renderer/commit/fecb2c05ba049ab096323fea46d8fe18ef79641a))

## [0.5.2](https://github.com/OpenDisplay/odl-renderer/compare/odl-renderer-v0.5.1...odl-renderer-v0.5.2) (2026-04-12)


### Features

* add blue and green colors to named color palette ([2d24e0e](https://github.com/OpenDisplay/odl-renderer/commit/2d24e0e0fe3c21c2b6dfb44067a5fc7e811fa0ea))
* **icons:** include mdi icons ([596d9b7](https://github.com/OpenDisplay/odl-renderer/commit/596d9b785186ffa56befa3508df46080d788ac7e))
* initial port ([b8d69ba](https://github.com/OpenDisplay/odl-renderer/commit/b8d69bad01300b55bd2999532665a0e71a50bfbf))
* **text:** support hex color tags in parse_colors ([aab85c3](https://github.com/OpenDisplay/odl-renderer/commit/aab85c35c75ff9c585ffe2fd0db4765257c272a8)), closes [#4](https://github.com/OpenDisplay/odl-renderer/issues/4)
* **visualizations:** add plot draw type ([7b27be2](https://github.com/OpenDisplay/odl-renderer/commit/7b27be21da66a9116caa7f7f95ead52b7ebf86b7))


### Bug Fixes

* remove pytest from normal dependencies ([e016177](https://github.com/OpenDisplay/odl-renderer/commit/e0161773d26d917b28369bd608ddcf2eb5c6869c))
* **text:** parse blue/green color tags ([65348e3](https://github.com/OpenDisplay/odl-renderer/commit/65348e3fbb357b42bac4c485f6b95787ec31dee3))


### Documentation

* add badges ([1ae2ff1](https://github.com/OpenDisplay/odl-renderer/commit/1ae2ff1960c7f011f985e6c98d7b6b37082c55c7))
* change json to python ([852610e](https://github.com/OpenDisplay/odl-renderer/commit/852610e4941222f3835ca2c6c8597b111c83fde2))
* update readme ([7cb40c0](https://github.com/OpenDisplay/odl-renderer/commit/7cb40c04e7d2d42bbb01389981c2a58ff02af9fa))


### Code Refactoring

* rename package from drawcustom to odl-renderer ([3573e49](https://github.com/OpenDisplay/odl-renderer/commit/3573e498e1a5cd59ab7e73d4f8af34b8a2fbe829))


### Tests

* add forgotten plot rendering ([8203094](https://github.com/OpenDisplay/odl-renderer/commit/8203094afdda5e0c34514da62e9cef1147588a07))
* add forgotten snapshots ([771ee87](https://github.com/OpenDisplay/odl-renderer/commit/771ee8786508a3e34a9a499a74fb8b25f60f1150))
* add integration tests for shapes, debug, text, and icons ([a7a54f9](https://github.com/OpenDisplay/odl-renderer/commit/a7a54f9ec533e86a94855d096d511d27a46e6c63))
* add test suite ([f07311c](https://github.com/OpenDisplay/odl-renderer/commit/f07311caaab77cb7eee51c36332426a5ec9349fa))
* add tests for media_loader, fonts, coordinates, core, media, visualizations ([dccd47d](https://github.com/OpenDisplay/odl-renderer/commit/dccd47dab3254eea95ea5a0c08aa6642b29a2358))
* fix dependencies ([f54901f](https://github.com/OpenDisplay/odl-renderer/commit/f54901fec0f96ff615058c1be94aba1de106380f))
* **plot:** add integration tests for plot element ([b0c58cb](https://github.com/OpenDisplay/odl-renderer/commit/b0c58cbfae0cd12d7558b71c646d029cf2baafcc))
* remove redundant visual-tests workflow step ([7978b6d](https://github.com/OpenDisplay/odl-renderer/commit/7978b6dc0892cbd6eb39a18cfdb0c80a2f6449ac))
* skip visual tests in CI ([ef4814d](https://github.com/OpenDisplay/odl-renderer/commit/ef4814d9c241efb90d97156785f4e5ba027bc75d))
* skip visual tests in CI coverage ([93046df](https://github.com/OpenDisplay/odl-renderer/commit/93046df78f1bdd51ed23b5586739c02cf8288721))


### Continuous Integration

* add cache to lint workflow ([061b421](https://github.com/OpenDisplay/odl-renderer/commit/061b421309f09a724ccf155e610de667ede73cec))
* add cache to test workflow ([040113a](https://github.com/OpenDisplay/odl-renderer/commit/040113aaa4cf81c5a9ba87149be4645669a760f7))
* add python 3.14 to test matrix ([3c7169d](https://github.com/OpenDisplay/odl-renderer/commit/3c7169d3150e1f1279e3260200b8cb4d1bfbff5a))
* add separate token again ([3e6fcaa](https://github.com/OpenDisplay/odl-renderer/commit/3e6fcaa0cf9b59aa7404fcd942aca4164da50bb0))
* add token to release workflow for authentication ([4ad11e9](https://github.com/OpenDisplay/odl-renderer/commit/4ad11e9232964cb5c01c3977b654cd97be6f4863))
* disable mypy for now ([55293d5](https://github.com/OpenDisplay/odl-renderer/commit/55293d5dd0974b472c7a2f7c55eb3fd887d2007b))
* remove extra-files section from release config ([d09aa29](https://github.com/OpenDisplay/odl-renderer/commit/d09aa2982e96c01fbd3432f9a0bafade7b8c0d8b))
* remove separate token ([b0700e0](https://github.com/OpenDisplay/odl-renderer/commit/b0700e049e83a5a769dab5e7e957eb70270fc252))
* remove unnecessary cache ([eb280ce](https://github.com/OpenDisplay/odl-renderer/commit/eb280ce5f5a8c39fd8f9cbcc6f5e6af693f4cf3a))
* separate lint and test workflows ([7a043d3](https://github.com/OpenDisplay/odl-renderer/commit/7a043d344a5804ee451f20e393d293ee64eda95f))
* update cache ([522a9b1](https://github.com/OpenDisplay/odl-renderer/commit/522a9b1bfbab3c77f26b52d7fc74356f4245ed22))
* update ci ([684fa56](https://github.com/OpenDisplay/odl-renderer/commit/684fa5662079f4423a6f692f0dae1ece5eed6724))
* update lint workflow ([fd4e614](https://github.com/OpenDisplay/odl-renderer/commit/fd4e614922a5728cdbc1455ce6fccc9dafa7716a))
* update test workflow ([52b880e](https://github.com/OpenDisplay/odl-renderer/commit/52b880ef37aab88546df1b94677f0a8c499d6c0d))

## [0.5.1](https://github.com/g4bri3lDev/drawcustom/compare/v0.5.0...v0.5.1) (2026-04-12)


### Bug Fixes

* remove pytest from normal dependencies ([e016177](https://github.com/g4bri3lDev/drawcustom/commit/e0161773d26d917b28369bd608ddcf2eb5c6869c))

## [0.5.0](https://github.com/g4bri3lDev/drawcustom/compare/v0.4.0...v0.5.0) (2026-04-10)


### Features

* **visualizations:** add plot draw type ([7b27be2](https://github.com/g4bri3lDev/drawcustom/commit/7b27be21da66a9116caa7f7f95ead52b7ebf86b7))

## [0.4.0](https://github.com/g4bri3lDev/drawcustom/compare/v0.3.0...v0.4.0) (2026-01-16)


### Features

* **icons:** include mdi icons ([596d9b7](https://github.com/g4bri3lDev/drawcustom/commit/596d9b785186ffa56befa3508df46080d788ac7e))

## [0.3.0](https://github.com/g4bri3lDev/drawcustom/compare/v0.2.0...v0.3.0) (2026-01-15)


### Features

* **text:** support hex color tags in parse_colors ([aab85c3](https://github.com/g4bri3lDev/drawcustom/commit/aab85c35c75ff9c585ffe2fd0db4765257c272a8)), closes [#4](https://github.com/g4bri3lDev/drawcustom/issues/4)


### Bug Fixes

* **text:** parse blue/green color tags ([65348e3](https://github.com/g4bri3lDev/drawcustom/commit/65348e3fbb357b42bac4c485f6b95787ec31dee3))

## [0.2.0](https://github.com/g4bri3lDev/drawcustom/compare/v0.1.1...v0.2.0) (2025-12-28)


### Features

* add blue and green colors to named color palette ([2d24e0e](https://github.com/g4bri3lDev/drawcustom/commit/2d24e0e0fe3c21c2b6dfb44067a5fc7e811fa0ea))


### Documentation

* add badges ([1ae2ff1](https://github.com/g4bri3lDev/drawcustom/commit/1ae2ff1960c7f011f985e6c98d7b6b37082c55c7))

## [0.1.1](https://github.com/g4bri3lDev/drawcustom/compare/v0.1.0...v0.1.1) (2025-12-28)


### Documentation

* change json to python ([852610e](https://github.com/g4bri3lDev/drawcustom/commit/852610e4941222f3835ca2c6c8597b111c83fde2))

## 0.1.0 (2025-12-28)


### Features

* initial port ([b8d69ba](https://github.com/g4bri3lDev/drawcustom/commit/b8d69bad01300b55bd2999532665a0e71a50bfbf))
