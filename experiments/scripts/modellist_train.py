
import matplotlib.pyplot as plt
import torch
from torch.nn import ModuleList

import gpytorch


from stancemining.estimate import get_classifier_profiles, GPClassificationModel, NormalPrior, BoundedOrdinalWithErrorLikelihood

def get_modellist_predictions(models, all_classifier_ids):
    model = gpytorch.models.IndependentModelList(*models)
    likelihood = LikelihoodList(*[m.likelihood for m in models])

    mll = SumVariationalELBO(likelihood, model)

    model.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    training_iterations = 50
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets, classifier_ids=all_classifier_ids)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    
    # Make predictions (use the same test points)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_xs = []
        test_classifier_ids = []
        for i in range(len(model.models)):
            test_x = torch.linspace(model.models[i].train_inputs.min(), model.models[i].train_inputs.max(), 51)
            test_xs.append(test_x)
        # This contains predictions for both outcomes as a list
        predictions = model(*test_xs)

    return predictions, model.models, test_xs

def get_predictions(models, all_classifier_ids):
    predictions = []
    test_xs = []
    for model, classifier_ids in zip(models, all_classifier_ids):
        mll = VariationalELBO(model.likelihood, model, model.train_targets.size(0))

        model.train()
        model.likelihood.train()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        training_iterations = 50
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(model.train_inputs)
            loss = -mll(output, model.train_targets, classifier_ids=classifier_ids)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

        # Set into eval mode
        model.eval()
        model.likelihood.eval()

        
        # Make predictions (use the same test points)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(model.train_inputs.min(), model.train_inputs.max(), 51)
            test_xs.append(test_x)
            # This contains predictions for both outcomes as a list
            prediction = model(test_x)
        predictions.append(prediction)
    return predictions, models, test_xs

def main():
    lengthscale_loc = 2.0
    lengthscale_scale = 0.5
    sigma_loc = 1.0
    sigma_scale = 0.1

    classifier_profiles = get_classifier_profiles()
    models = []
    all_classifier_ids = []
    num_models = 3
    for _ in range(num_models):
        train_x = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32)
        a = torch.rand(()) * 0.5 + 0.5
        b = torch.rand(()) * 0.5 + 0.5
        train_y = torch.round(torch.sin(a * train_x + b) + torch.normal(torch.zeros_like(train_x), 0.1 * torch.ones_like(train_x))) + 1
        classifier_ids = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int64)

        bin_edges = torch.tensor([-0.5, 0.5])
        # likelihood = OrdinalLikelihood(bin_edges)
        sigma_prior = NormalPrior(sigma_loc, sigma_scale)
        likelihood = BoundedOrdinalWithErrorLikelihood(bin_edges, classifier_profiles, sigma_prior=sigma_prior)
        
        max_inducing_points = 1000
        if train_x.size(0) > max_inducing_points:
            learn_inducing_locations = True
            perm = torch.randperm(train_x.size(0))
            idx = perm[:max_inducing_points]
            inducing_points = train_x[idx]
        else:
            learn_inducing_locations = False
            inducing_points = train_x

        model = ApproximateGPModelListModel(
            inducing_points, 
            learn_inducing_locations=learn_inducing_locations, 
            lengthscale_loc=lengthscale_loc, 
            lengthscale_scale=lengthscale_scale,
            variational_dist='natural',
            train_inputs=train_x,
            train_targets=train_y,
            likelihood=likelihood
        )
        models.append(model)
        all_classifier_ids.append(classifier_ids)

    # Initialize plots
    f, axs = plt.subplots(2, num_models, figsize=(10, 6))

    ml_predictions, ml_models, ml_test_xs = get_modellist_predictions(models, all_classifier_ids)
    predictions, models, test_xs = get_predictions(models, all_classifier_ids)

    axs = axs.flatten()

    for submodel, prediction, test_x, ax in zip(ml_models + models, ml_predictions + predictions, ml_test_xs + test_xs, axs):
        mean = prediction.loc
        lower, upper = prediction.confidence_region()

        mean, lower, upper = torch.tanh(mean), torch.tanh(lower), torch.tanh(upper)

        tr_x = submodel.train_inputs.detach().numpy()
        tr_y = submodel.train_targets.detach().numpy() - 1

        # Plot training data as black stars
        ax.plot(tr_x, tr_y, 'k*')
        # Predictive mean as blue line
        ax.plot(test_x.numpy(), mean.numpy(), 'b')
        # Shade in confidence
        ax.fill_between(test_x.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(f"Learned Lengthscale: {submodel.covar_module.base_kernel.lengthscale.item():.2f}")
    f.savefig('./figs/modellist_train.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()