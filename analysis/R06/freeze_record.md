| item | sha256 / value |
|---|---|
| `build_inputs.sbatch` | `04a86643da7330bd5c080a2cd568dbeb8824b3ebe9f8c6c4b6add58dbd23e92f` |
| `build_inputs_r06.py` | `393c6dda7f29155a95af0cab2591a190042fa24c1164ede6adbbd3e4e66f2161` |
| `freeze_r06.py` | `fb207a35f3e38a7addeace63e6fa8cb85a2fde54d76afcec3c854e8d05ce2dee` |
| `make_tasks.py` | `22729a6a60b57e8c7bb83ca4c3f06547433d1931fb3f522ac1dec4c130d46bbb` |
| `pilot.sbatch` | `980d36b8034d7ed66163fc608a6f9ac6670903ed25cb0b4159a5ece1c2cc15ee` |
| `pilot_extrapolate.py` | `9657d4860317c1bf351db3e6afdd4cb988c06a2d49ef6c7d7aa98167657f33ee` |
| `preserve_r06.sbatch` | `ab667e25eb165e3def0725516fe0596ac270bcc48ff4ca7d2cb3dd08171e787d` |
| `r06_common.py` | `418b49fdc95ed295a75a01b934eed0cb57625b01127966be2addd30bd294910f` |
| `r06_model.py` | `8083d5b1346c1104eb07c9b65d0d37a88cc7edde9189ba672f45251957a33ec3` |
| `score_r06.py` | `76f3a23368afb80e49d3b6600c56c62a7efaf798e8da5f8412dfe42b47d58991` |
| `tasks.txt` | `9e3d727c4687a1c77eab43df72d7fff8c0049183496e123ddd7d1e966bcaf229` |
| `test_score_synthetic.py` | `94f604f2bd5df00f51486feacb79e7409f4922684a304ce48121a16907c8a0fa` |
| `train_array.sbatch` | `32dd359dd33044372bd2dd3b8ce543f53c4b3e00dd8a9d08b4909b841d87d510` |
| `train_r06.py` | `cc35be755ec1310e5b2376d0914a6e79061087906345b3c4152bc27dff4b5e58` |
| training code (r06_common + build_inputs_r06 + r06_model + train_r06) | `bf5a82b47fb95d01c66be3f718b3a239e319d1ef3b02fe35e8d51a3919316148` |
| training-code hashes found in the 100 train_log.json files | {'bf5a82b47fb95d01c66be3f718b3a239e319d1ef3b02fe35e8d51a3919316148': 100} |
| input leak test | {'n': 100, 'n_identical': 100, 'n_usable': 74} |
| phase0-freeze commit | 866320b7e0917460b63060e86b4a8be8b409c64e |
| git HEAD at run | a0947c803c8c22cdd99152c0094cdedbe4c6c26b |
| tag | phase2-r1 (applied to the R06 commit) |
