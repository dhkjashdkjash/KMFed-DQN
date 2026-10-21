KMFed-DQN
Research code for water-quality forecasting, ordinary and dueling deep Q-learning, and cost–risk Pareto analysis. Forecasting and control are separate workflows.

This distribution contains source code, examples, software tests, and documentation.

Installation and checks
The pinned dependencies have been tested with Python 3.11. Use Python 3.11 for this environment; the pinned NumPy and SciPy versions should not be installed directly into Python 3.14.

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
Data
Keep lawfully obtained, source-verified files outside this directory and provide them through or an explicit input-file path. The author-designated source entry points are:--data-dir

China National Environmental Monitoring Centre
Global River Water Quality Archive, Zenodo record 15335450
These entry points alone do not establish the provenance, quality-control status, or processing history of an individual record. Required fields, units, and provenance records are described in data/README.md.

Forecasting
python examples/01_kalman_state_estimation.py --data-dir /path/to/verified_stations --output-dir /path/to/derived_outputs
python examples/02_federated_forecasting.py --data-dir /path/to/verified_stations --train-end YYYY-MM-DD --validation-end YYYY-MM-DD --mode fedavg --output-dir /path/to/derived_outputs
The study design must specify one shared pair of calendar boundaries for all stations. The furthest training target must precede . Validation origins must be at or after , and their furthest targets must precede . Test origins must be at or after . Windows near these boundaries may therefore be excluded.train-endtrain-endvalidation-endvalidation-end

Windows require adjacent observations exactly four hours apart. Filtering restarts at gaps without interpolation across them. Forecast horizons are 4, 8, 12, 16, 20, 24, and 48 hours. The mode derives aggregation weights only from validation errors; the test set is reserved for final evaluation. Without a risk configuration, only regression metrics are calculated and is .qualitymacro_f1null

To evaluate warnings, supply . This JSON must contain a nonempty and a entry for each of the five indicators, with nested , , and intervals. Each interval is ; denotes an unbounded side. A zero lower concentration bound must not be treated as evidence of a pollution hazard at low concentrations. Threshold sources and applicability require verification; a source string is not proof of validation.--risk-config /path/to/risk.jsonsourcesensorsnormal_rangewarning_thresholdcritical_threshold[lower, upper]null

The baseline risk level is the maximum indicator level on a scale of 0–3, not a statutory water-quality classification. Its warning event occurs when any of the next six four-hour targets reaches level 2. Observed and predicted labels use the same definition. Events between sampling times are outside this evaluation's coverage.

Outputs retain unconstrained values as .prediction_raw

A training indicator with zero standard deviation uses 1 as the normalization denominator and is named in the output metadata. Its error cannot then be interpreted as normalized by observed variability.

For the baseline Kalman filter, is an algorithmic hyperparameter; and are scale conventions. Hyperparameters should be determined within the training and validation partitions. Filter outputs retain both unconstrained estimates and projection flags. A centered smoother over the full series is not used to construct forecasting inputs.q_over_r=0.01R=1P0=R·I

Example 02 exports preprocessing metadata, weights, validation history, independent test metrics, and a prediction-level table. Inference without future labels is available through:

from kmfed_dqn.fed_forecast import forecast_latest

prediction_table = forecast_latest('/path/to/forecaster_fedavg.npz', history)
history must be a verified, time-ordered DataFrame containing all five indicators. Model files support inference. Training-input fingerprints support traceability but do not certify data authenticity.

Control
Example 03 uses an ordinary DQN with two hidden layers, experience replay, and a target network. Its observations contain the current five indicators and their first differences. is an optional architecture; the trainer and target network preserve its value and advantage branches.DuelingDQN

Centralized training retains replay, target-update counters, and exploration state across stations. FedAvg weights clients by their actual environment-step counts and evaluates the aggregated model at each validation round. matches environment steps per station across paradigms; it does not establish equal optimizer-update or communication budgets.fed_control.get_paradigm_agents

python examples/03_train_control_agents.py --data-dir /path/to/verified_stations --control-module /path/to/calibrated_model.py --train-end YYYY-MM-DD --validation-end YYYY-MM-DD --output-dir /path/to/derived_outputs
The external Python module is executable code and must be trusted. It must define returning with:make_control_spec()kmfed_dqn.control_env.ControlSpec

action_names and : action labels and the no-intervention ID, with identical meanings across clients.no_action
costs and : a cost for each action at each time step, in consistent units.cost_unit
risk_penalties: nonnegative decision penalties for risk classes 0–3 and a rationale for their selection.
step_hours, , and : the validated model time step, source description, and .sourceriskRiskSpec
transition(state, observed_now, observed_next, action, dt_hours): an externally validated transition model returning a vector of five indicators. A stateful transition object should implement and document why the observation represents sufficient state information for training.reset()
Credible intervention conclusions require an explicitly specified response model and supporting evidence. No built-in purification ratios, dosing amounts, or electricity prices substitute for those inputs. Response models require separate validation across time scales. Control metrics describe simulated trajectories under that supplied model, rather than measured intervention effects. Evaluation uses up to three fixed windows per station, which can overlap and do not cover the entire test period. Invalid transition outputs stop execution rather than being clipped into usable observations.

Descriptive recovery time includes only events with fully observed onset and recovery, with left- and right-censored events counted separately. It is not a censoring-adjusted population estimate. Unjustified PID, preventive low-concentration intervention, and simplified greedy baselines are not included.

Checkpoints store architecture, learning rate, normalization, control specifications, time boundaries, and input fingerprints. They are inference files, not resumable training states with replay and optimizer history. Schema 3 supports ordinary and dueling networks; the loader also accepts schema 2 ordinary-network checkpoints. Older weights without the required metadata are incompatible. There is no automatic checkpoint reuse.

Cross-basin forecasting and policy tables
python examples/04_cross_basin_transfer.py /path/to/monthly.csv --output-dir /path/to/derived_outputs
python examples/05_policy_frontier.py /path/to/evaluated_policies.csv --output-dir /path/to/derived_outputs
Example 05 accepts already evaluated policies. It does not infer intervention outcomes from observed concentrations. Required fields are . Each basin–policy combination must have one row. Compared policies within a basin must share cost units, a risk definition, an aggregation method, and a time step.basin,policy_id,family,cost,risk,evaluation_source,cost_unit,risk_definition,aggregation,time_step

Pareto analysis retains ties with identical costs and risks. The label “Balanced” identifies the point nearest the ideal point in normalized two-dimensional space.

To construct monthly inputs, use with ; timestamps must use the declared study timezone. The function retains all months with an flag. Select eligible months explicitly before modeling. Months below the minimum count or without enough observations for the chosen variance estimator are not silently dropped. Duplicate observations, mixed units, and missing values raise errors. Input observations are not modified.aggregate_monthlybasin,station,param,time,value,uniteligible

For policy aggregation, requires . It aggregates externally calculated costs and residual risks. It does not identify intervention effects or estimate confidence intervals.aggregate_policy_evaluationsbasin,policy_id,station,month,family,cost,risk,evaluation_source,cost_unit,risk_definition,time_step

Performance and uncertainty claims require verified observations, a specified analysis plan, and the corresponding experiment outputs. The supplied software tests do not establish predictive accuracy or intervention effectiveness.

License
The source code is provided under the MIT License. Data use remains subject to the original providers' terms.
