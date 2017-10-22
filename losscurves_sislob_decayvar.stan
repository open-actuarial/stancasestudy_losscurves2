data {
    int<lower=0,upper=1> growthmodel_id;

    int n_data;
    int n_time;
    int n_cohort;

    int cohort_id[n_data];
    int t_idx[n_data];

    int cohort_maxtime[n_cohort];

    vector<lower=0>[n_time] t_value;

    vector[n_cohort] premium;
    vector[n_data]   loss;
}

parameters {
    real<lower=0> omega;
    real<lower=0> theta;

    real<lower=0> var_decayrate;

    vector<lower=0>[n_cohort] LR;

    real mu_LR;
    real<lower=0> sd_LR;

    real<lower=0> loss_sd;
}

transformed parameters {
    vector[n_time] gf;
    vector[n_time] decay_prop;

    vector[n_data] d_lm;
    vector[n_data] d_sd;

    for(i in 1:n_time) {
        gf[i] = growthmodel_id == 1 ?
            weibull_cdf (t_value[i], omega, theta) :
            logistic_cdf(t_value[i], omega, theta);

        decay_prop[i] = exp(-var_decayrate * t_value[i]);
    }

    for (i in 1:n_data) {
        d_lm[i] = LR[cohort_id[i]] * premium[cohort_id[i]] * gf[t_idx[i]];
        d_sd[i] = premium[cohort_id[i]] * loss_sd * decay_prop[t_idx[i]];
    }
}

model {
    mu_LR ~ normal(0, 0.5);
    sd_LR ~ lognormal(0, 0.5);

    var_decayrate ~ lognormal(0, 0.3);

    LR ~ lognormal(mu_LR, sd_LR);

    loss_sd ~ lognormal(0, 0.7);

    omega ~ lognormal(0, 0.5);
    theta ~ lognormal(0, 0.5);

    loss ~ normal(d_lm, d_sd);
}


generated quantities {
}
