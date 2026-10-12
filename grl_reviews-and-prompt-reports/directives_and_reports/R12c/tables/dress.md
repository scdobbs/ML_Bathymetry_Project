<!-- generated from tables/dress_check.json (check_dress_r12c.py) -->
| step | rc | PermissionError lines | tracebacks | warning categories | mock guard: before refused / after allowed / FAKE refused / real sealed refused | tokens remaining |
|---|---|---|---|---|---|---|
| build (`dress_build_47546691.log`) | 0 | 0 | 0 | {'ResourceWarning': 42, 'DeprecationWarning': 144} | True / True / True / True | 37 (the 37 real sealed tokens) |
| figures (`dress_figures_47546691.log`) | 0 | 0 | 0 | {'ResourceWarning': 44, 'DeprecationWarning': 4, 'RuntimeWarning': 6} | True / True / True / True | 37 (the 37 real sealed tokens) |
| predict (`dress_predict_47546691.log`) | 0 | 0 | 0 | {'ResourceWarning': 48, 'FutureWarning': 3} | True / True / True / True | 37 (the 37 real sealed tokens) |
| score (`dress_score_47546691.log`) | 0 | 0 | 0 | {'ResourceWarning': 38} | True / True / True / True | 37 (the 37 real sealed tokens) |
| FK151121__MGDS_22383 | built, products used | tiles 732, D₁ 66 px, interior 523 / 50 blocks, view interior | predictions finite {'s_lr': 732, 'GBT_ship': 732, 'P-V1': 732} of 732 | score: {'view': 'interior', 'status': 'in C1', 'n_scored': 523, 'ci_estimable': True} | | |
| TN399__MGDS_33068 | built, products masked | tiles 178, D₁ 45 px, interior 1 / 1 blocks, view fallback | predictions finite {'s_lr': 178, 'GBT_ship': 178, 'P-V1': 178} of 178 | score: {'view': 'fallback', 'status': 'in C1', 'n_scored': 167, 'ci_estimable': True} | | |
| KM0923__MGDS_31429 | built, products used | tiles 223, D₁ 43 px, interior 24 / 8 blocks, view interior | predictions finite {'s_lr': 223, 'GBT_ship': 223, 'P-V1': 223} of 223 | score: {'view': 'interior', 'status': 'in C1', 'n_scored': 24, 'ci_estimable': True} | | |
| figure set A | 3 pages, codes ['Q01', 'Q02', 'Q03'], text objects per page [0, 0, 0] | | | | | |
| EW0207 control | job 47546691, sh03-09n34.int (AMD EPYC 7543 32-Core Processor) | s_lr identical True | GBT 2.22e-16 m | P-V1 1.79e-07 | passed **True** | |
| real sealed paths opened (any script) | 0 | | | | | |
