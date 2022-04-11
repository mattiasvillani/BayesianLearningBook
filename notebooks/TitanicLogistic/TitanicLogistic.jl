using Plots, Distributions, GLM, LinearAlgebra, Optim, ForwardDiff, Utils, CSV
using KernelDensity, StatsPlots, Measures, AdvancedHMC

bookFolder = "/home/mv/Dropbox/BayesBook/"
figFolder = bookFolder*"Figs/"
dataFolder = bookFolder*"Data/"
giturl = "https://github.com/mattiasvillani/BayesianLearningBook/"

gr(legend = nothing, grid = false, color = colors[2], lw = 3, legendfontsize=10,xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14, markersize = 0)

    
""" 
    πlogisticreg(β, y, X, μ, Σ) 

log posterior for the logistic regression model
        Pr(y=1|x) = 1/(1 + exp(-x'β)) 
with the prior 
        β ~ N(μ,Σ).
""" 
function πlogisticreg(β, y, X, μ, Σ)                        
    loglik = sum( y.*(X*β)  .- log.(1 .+ exp.(X*β)) )
    logprior = logpdf(MvNormal(μ, Σ), β)
    return(loglik + logprior)
end

logistic(x) = 1/(1 + exp(-x))

# titanic data
data = CSV.read(dataFolder*"titanic.csv", DataFrame)
n = size(data,1)
y = data.survived
X = [ones(n,1) data.age data.sex .=="female" data.pclass .== 1]
#rename!(X,["intercept","age","sex","class"])
varnames = ["intercept","age","sex","class"]

# Noninformative prior
p = size(X,2)
μ = zeros(p)
Σ = 10^2*I(p)

# implied prior on OR for sex and first class
quantiles = [quantile(LogNormal(μ[j], √Σ[j,j]),[0.025,0.975])' for j = 1:4]

# Run optimizer
glmfit = glm(X, y, Bernoulli(), LogitLink()) # find MLE.
β₀ = coef(glmfit) # initial values from MLE.
optres = maximize(β -> πlogisticreg(β, y, X, μ, Σ), β₀, autodiff = :forward)
βmode = Optim.maximizer(optres)
H(β) = ForwardDiff.hessian(β -> πlogisticreg(β, y, X, μ, Σ), β)
Ωᵦ = Symmetric(-inv(H(βmode))) 

p = []
for i = 1:4
    lognormpdf(x) = pdf(LogNormal(βmode[i], √Ωᵦ[i,i]),x)
    quantiles = quantile(LogNormal(βmode[i], √Ωᵦ[i,i]),[0.0001,0.9999])
    push!(p, plot(lognormpdf, xlims = (quantiles[1],quantiles[2]), c = colors[2], 
    title = varnames[i], xlab = L"\exp(\beta_{%$(i-1)})", yticks = [], yaxis = false, 
    lw = 3, margin = 0mm))
end
plot(p..., size = (600,600))
savefig(figFolder*"titanic_post_oddsratio_noninfo.pdf")


# Joint baseline odds and oddsratio for age
ρ, σ = Cov2Corr(Ωᵦ)
βsim = rand(MvNormal(βmode,Ωᵦ), 10000)'
oddsratio = exp.(βsim) # 10000 × 4 matrix with draws of exp(βⱼ) in jth column.
#kdejoint = kde(oddsratio[:,1:2])
#plot(kdejoint, lw = 1)
scatter(oddsratio[:,1], oddsratio[:,2], markersize = 2, markerstrokecolor = :auto,
    color = colors[2], xlab = L"\exp(\beta_0)", ylab = L"\exp(\beta_1)", margin = 3mm)
savefig(figFolder*"titanic_joint_oddsratio_intercept_age.png")

# INFORMATIVE PRIOR
μ = [-1,-1/80,1,1]
Σ = diagm([0.25,1/(80^2),0.5,1])

# implied prior on OR for sex and first class
lognormpdf(x) = pdf(LogNormal(μ[3], √Σ[3,3]),x)
quantiles = quantile(LogNormal(μ[3], √Σ[3,3]),[0.025,0.97])
plot(lognormpdf, xlims = (0,quantiles[2]), c = colors[2], 
    title = "$(varnames[3]) and $(varnames[4])", xlab = "survival odds ratio", yticks = [], yaxis = false, margin = 7mm, label = L"\mathrm{sex}", legend = :topright)
lognormpdf(x) = pdf(LogNormal(μ[4], √Σ[4,4]),x)
quantiles = quantile(LogNormal(μ[4], √Σ[4,4]),[0.025,0.97])
plot!(lognormpdf, c = colors[4], label = L"\mathrm{class}")
savefig(figFolder*"titanic_prior_sex_class.pdf")

# Prior for age β
agegrid = [0,40,80]
Xgridage = [ones(3,1) agegrid zeros(3,2)] 
means = Xgridage*μ
stds = .√diag(Xgridage*Σ*Xgridage')
dists = LogNormal.(means, stds)
h = plot(legend = :right, legendfontsize = 12, xlab = "survival odds for male in 2nd or 3rd class",
    yticks = [], yaxis = false, xlims = (0,1), title = "age", margin = 7mm)
for j = 1:length(means)
    plot!(h, 0:0.01:10, pdf.(dists[j],0:0.01:10), color = colors[j],
        label = L"\mathrm{age}=%$(agegrid[j])")
end
h
savefig(figFolder*"titanic_prior_age.pdf")


# implied prior on OR
quantiles = [quantile(LogNormal(μ[j], √Σ[j,j]),[0.025,0.975])' for j = 1:4]

# Run optimizer
glmfit = glm(X, y, Bernoulli(), LogitLink()) # find MLE.
β₀ = coef(glmfit) # initial values from MLE.
optres = maximize(β -> πlogisticreg(β, y, X, μ, Σ), β₀, autodiff = :forward)
βmodeInfo = Optim.maximizer(optres)
H(β) = ForwardDiff.hessian(β -> πlogisticreg(β, y, X, μ, Σ), β)
ΩᵦInfo = Symmetric(-inv(H(βmodeInfo))) 

h = []
labels = ["noninformative prior" "informative prior";nothing nothing;nothing nothing;nothing nothing];
for i = 1:4
    lognormpdf(x) = pdf(LogNormal(βmode[i], √Ωᵦ[i,i]),x)
    quantiles = quantile(LogNormal(βmode[i], √Ωᵦ[i,i]),[0.0001,0.9999])
    ptmp = plot(lognormpdf, xlims = (quantiles[1],quantiles[2]), c = colors[2], 
        title = varnames[i], xlab = L"\exp(\beta_{%$(i-1)})", yticks = [], yaxis = false, 
        lw = 3, margin = 0mm, label = labels[i,1])
    lognormpdf(x) = pdf(LogNormal(βmodeInfo[i], √ΩᵦInfo[i,i]),x)
    plot!(ptmp, lognormpdf, xlims = (quantiles[1],quantiles[2]), c = colors[4], 
        title = varnames[i], xlab = L"\exp(\beta_{%$(i-1)})", yticks = [], yaxis = false, 
        lw = 3, margin = 0mm, label = labels[i,2])
    push!(h, ptmp)
end
plot(h..., size = (600,600), legend = :topright)
savefig(figFolder*"titanic_post_oddsratio_info.pdf")

# HMC
ℓπ(β) = πlogisticreg(β, y, X, μ, Σ)
n_samples, n_adapts = 100_000, 1_000
hamiltonian = Hamiltonian(DiagEuclideanMetric(p), ℓπ, ForwardDiff)
initial_ϵ = find_good_stepsize(hamiltonian, β₀)
integrator = Leapfrog(initial_ϵ)
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
samples, stats = sample(hamiltonian, proposal, β₀, n_samples, adaptor, 
    n_adapts; progress=true);
samples = reduce(hcat,samples)'
oddsratio_samples = exp.(samples)

h = []
labels = ["simulated" "normal approx" "MLE";nothing nothing nothing;nothing nothing nothing;nothing nothing nothing];
for i = 1:4
    lognormpdf(x) = pdf(LogNormal(βmodeInfo[i], √ΩᵦInfo[i,i]),x)
    quantiles = quantile(LogNormal(βmode[i], √Ωᵦ[i,i]),[0.0001,0.9999])
    ptmp = histogram(oddsratio_samples[:,i], nbins = 100, linecolor = nothing, 
        normalize = true, xlims = (quantiles[1],quantiles[2]), title = varnames[i], xlab = L"\exp(\beta_{%$(i-1)})", yticks = [], yaxis = false, margin = 0mm, 
        color = colors[1], label = labels[i,1])    
    plot!(ptmp, lognormpdf, c = colors[2], label = labels[i,2], lw = 2)
    scatter!([exp(β₀[i])],[0], color = colors[4], markersize = 6, markerstrokecolor = :auto, label = labels[i,3])
    push!(h, ptmp)
end
plot(h..., size = (600,600), legend = :topright)
savefig(figFolder*"titanic_post_oddsratio_info_vs_hmc.pdf")

# AGE AND FARE ONLY
data = DataFrame(CSV.File(dataFolder*"titanic.csv"))
n = size(data,1)
y = data.survived
X = DataFrame([ones(n,1) data.age data.fare], :auto)
rename!(X,["intercept","age","fare"])


covGLM = vcov(glmfit) 

""" 
    πprobitreg(β, y, X, μ, Σ) 

log posterior for probit regression Pr(y=1|x) = Φ(x'β) with prior β ~ N(μ,Σ).
""" 
function πprobitreg(β, y, X, μ, Σ)                         
    loglik = sum(y.*logcdf(Normal(), X*β) + (1 .- y).*logccdf(Normal(), X*β))
    if isinf(loglik) loglik = eps() end
    logprior = logpdf(MvNormal(μ, Σ), β)
    return(loglik + logprior)
end