# Linear regression for RDataset Salaries from the car package
# For the book Bayesian Learning by Mattias Villani

using Plots, Distributions, LinearAlgebra, CSV, HTTP, DataFrames, Latexify, RDatasets, LaTeXStrings
import ColorSchemes: Paired_12; colors = Paired_12[[1,2,7,8,3,4,5,6,9,10,11,12]]
using Flux: onehot, onehotbatch

bookFolder = "/home/mv/Dropbox/BayesBook/"
figFolder = bookFolder*"Figs/"
dataFolder = bookFolder*"Data/"
giturl = "https://github.com/mattiasvillani/BayesianLearningBook/"

include(bookFolder*"Code/Linreg.jl") # Functions for Bayesian linear regression
include(bookFolder*"Code/Distr.jl") # Functions for Bayesian linear regression

giturl = "https://github.com/mattiasvillani/BayesianLearningBook/raw/main/"
gr(legend = nothing, grid = false, color = colors[2], lw = 2, legendfontsize=10,
	xtickfontsize=12, ytickfontsize=12, xguidefontsize=14, yguidefontsize=14, 
    markersize = 5, markerstrokecolor = :auto)
    

# Reading the univerisity salaries dataset from the RDatasets package
df = dataset("car","Salaries")
df.logsalary = log.(df.Salary)
df.phdage = df.YrsSincePhD ./ maximum(df.YrsSincePhD) # normalize academic age
df.phdagesqr = df.phdage.^2

# Create binary dummy variables for categorical covariates (one-hot encoding)
Z = onehotbatch(df.Rank, ["AsstProf", "AssocProf", "Prof"])'
for k ∈ 2:size(Z,2)
	df[!,Symbol("rank",k)] = Z[:,k]
end
df.sex = (df.Sex .== "Male")
df.discipline = (df.Discipline .== "A")

# Plot logsalary against phdage
scatter(df.phdage, df.logsalary, color = colors[2], 
    ylabel = "log salary", xlabel = "years since PhD (normalized)", label = "")
savefig(figFolder*"SalariesScatter.pdf")

# Plot logsalary against phdage by rank
sel = (df.Rank .=="AsstProf")
scatter(df[sel,:phdage], df[sel,:logsalary], color = colors[1], 
    ylabel = "log salary", xlabel = "years since PhD (normalized)", label = "Asst. Prof")
sel = df.Rank .=="AssocProf"
scatter!(df[sel,:phdage], df[sel,:logsalary], color = colors[2], label = "Assoc. Prof")
sel = df.Rank .=="Prof"
scatter!(df[sel,:phdage], df[sel,:logsalary], color = colors[3], label = "Prof", legend=:topleft)
savefig(figFolder*"SalariesScatterByRank.pdf")


# Plot logsalary against phdage by sex
sel = (df.Sex .=="Male")
scatter(df[sel,:phdage], df[sel,:logsalary], color = colors[1], 
    ylabel = "log salary", xlabel = "years since PhD (normalized)", label = "Male")
sel = (df.Sex .=="Female")
scatter!(df[sel,:phdage], df[sel,:logsalary], color = colors[8], label = "Female", legend=:topleft)
savefig(figFolder*"SalariesScatterBySex.pdf")

# Plot logsalary against phdage by discipline
sel = (df.Discipline .=="A")
scatter(df[sel,:phdage], df[sel,:logsalary], color = colors[2], 
    ylabel = "log salary", xlabel = "years since PhD (normalized)", label = "Discipline A")
sel = (df.Discipline .=="B")
scatter!(df[sel,:phdage], df[sel,:logsalary], color = colors[3], label = "Discipline B", legend=:topleft)
savefig(figFolder*"SalariesScatterByDiscipline.pdf")

select!(df,[:logsalary,:phdage,:phdagesqr,:rank2,:rank3,:sex,:discipline]) # Drop unneccesary variables


# Simulate from σ² prior
nSim = 10000
igr(s²) = quantile(LogNormal(m,√s²),0.90) - quantile(LogNormal(m,√s²),0.10)
m = log(80000)
ν₀ = 10
σ²₀ = 0.3^2
σ²sim = rand(ScaledInverseChiSq(ν₀,σ²₀),nSim)

# Plotting the log normal distribution for salaries at σ²₀ 
gr(xtickfontsize=12, ytickfontsize=12, xguidefontsize=14, yguidefontsize=14)
salaryGrid = 0:100:250000
pdfGrid = pdf(LogNormal(m,√σ²₀),salaryGrid)
col = colors[2]
lower = quantile(LogNormal(m,√σ²₀),0.10)
upper = quantile(LogNormal(m,√σ²₀),0.90)
plot(salaryGrid, pdf(LogNormal(m,√σ²₀),salaryGrid), lw = 4, color = col, 
    xlabel = "Salary", yaxis = nothing, ylims = [], 
    xticks = (0:50000:250000, 0:50000:250000), label = "")
plot!([lower, upper], [0,0], color = colors[4], lw = 8, label = "", legend=:topright)
savefig(figFolder*"salaryimpliedlognormal.pdf")

# Plot prior for σ²
σGrid = 0:0.001:√0.4
plot(σGrid.^2, pdf(ScaledInverseChiSq(ν₀,σ²₀),σGrid.^2), lw = 4, xlabel = L"\sigma^2", 
    yaxis = nothing, color = colors[6])
savefig(figFolder*"salarypriorsigma2.pdf")

# Plot the difference between 90% and 10% percentiles
iqrdraws = igr.(σ²sim)
quantile(iqrdraws,0.025),quantile(iqrdraws,0.975)
histogram(iqrdraws, bins = 0:5000:200000, color = colors[2], linecolor=:transparent, yaxis = nothing,
    xticks = (0:50000:250000, 0:50000:200000), xlabel = "Salary spread")
savefig(figFolder*"salarypriorpercentilediff.pdf")


# Simulate curves from prior 
X = [ones(n,1) Matrix(df[!,Not(:logsalary)])]
μ₀ = [log(70000), 2, -1.5, 0, 0, 0, 0]
ν₀ = 10
σ²₀ = 0.3^2
xstar = 0:0.01:1
Xstar = [ones(length(xstar)) xstar xstar.^2]
nDraws = 10

# Simulate from prior κ₀ = 0.01
κ₀ = 0.01
Ω₀ = κ₀*(X'*X)
βsim, σ²sim  = BayesLinRegPrior(μ₀, Ω₀, ν₀, σ²₀, nSim)
h = plot(xstar, exp.(Xstar*μ₀[1:3])./1000, color = colors[6], lw = 4,
    title = L"\kappa_0 = 0.01", ylabel = "salary (in \$1000)", 
    xlabel = "years since PhD (normalized)")
for i = 1:nDraws
    plot!(h,xstar, exp.(Xstar*βsim[i,1:3])./1000, color = colors[5], lw = 2)
end
h
savefig(figFolder*"SalariesSimPriorCurvesKappa001.pdf")

# Simulate from prior κ₀ = 0.1
κ₀ = 0.1
Ω₀ = κ₀*(X'*X)
βsim, σ²sim  = BayesLinRegPrior(μ₀, Ω₀, ν₀, σ²₀, nSim)
h = plot(xstar, exp.(Xstar*μ₀[1:3])./1000, color = colors[6], lw = 4,
    title = L"\kappa_0 = 0.1", ylabel = "salary (in \$1000)", 
    xlabel = "years since PhD (normalized)")
for i = 1:nDraws
    plot!(h,xstar, exp.(Xstar*βsim[i,1:3])./1000, color = colors[5], lw = 2)
end
h
savefig(figFolder*"SalariesSimPriorCurvesKappa01.pdf")

# Simulate from prior κ₀ = 1
κ₀ = 1
Ω₀ = κ₀*(X'*X)
βsim, σ²sim  = BayesLinRegPrior(μ₀, Ω₀, ν₀, σ²₀, nSim)
h = plot(xstar, exp.(Xstar*μ₀[1:3])./1000, color = colors[6], lw = 4,
    title = L"\kappa_0 = 1", ylabel = "salary (in \$1000)", 
    xlabel = "years since PhD (normalized)")
for i = 1:nDraws
    plot!(h,xstar, exp.(Xstar*βsim[i,1:3])./1000, color = colors[5], lw = 2)
end
h
savefig(figFolder*"SalariesSimPriorCurvesKappa1.pdf")

# Simulate from prior κ₀ = 10
κ₀ = 10
Ω₀ = κ₀*(X'*X)
βsim, σ²sim  = BayesLinRegPrior(μ₀, Ω₀, ν₀, σ²₀, nSim)
h = plot(xstar, exp.(Xstar*μ₀[1:3])./1000, color = colors[6], lw = 4,
    title = L"\kappa_0 = 10", ylabel = "salary (in \$1000)", 
    xlabel = "years since PhD (normalized)")
for i = 1:nDraws
    plot!(h,xstar, exp.(Xstar*βsim[i,1:3])./1000, color = colors[5], lw = 2)
end
h
savefig(figFolder*"SalariesSimPriorCurvesKappa10.pdf")

# Posterior analysis
κ₀ = 1
Ω₀ = κ₀*(X'*X)
nSim = 10000
Xnames = ["intercept"]
Xnames = vcat(Xnames,names(df)[2:end])
μₙ, Ωₙ, νₙ, σ²ₙ, βsim, σ²sim = BayesLinReg(y, X, μ₀, Ω₀, ν₀, σ²₀, nSim)
    
stdPrior = .√((ν₀/(ν₀-2))*diag(σ²₀*inv(Ω₀))) # just checking
covPost = σ²ₙ*inv(Ωₙ)
stdPost = .√((νₙ/(νₙ-2))diag(covPost))

# Marginal posterior summary
p = size(X,2)
postSummary = zeros(p,4)
postSummary[:,1] = μₙ
postSummary[:,2] = stdPost
for j = 1:p
	postSummary[j,3:4] .= quantile(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),[0.025,0.975]) 
end
σ²margin = ScaledInverseChiSq(νₙ,σ²ₙ)
σmargin = .√σ²sim
σSummary = [mean(σmargin), std(σmargin), quantile(σmargin,0.025), quantile(σmargin,0.95)]
postSummary = [postSummary;σSummary']
push!(Xnames,"sigma")
postSummary = DataFrame(postSummary)
rename!(postSummary, [:mean,:std,:lower95,:upper95])
postSummaryLatex = latexify(postSummary, env = :tabular, fmt = "%.2f", side = Xnames)
    
# Marginal posterior densities
gr(legendfontsize=8, xtickfontsize=8, ytickfontsize=8, xguidefontsize=14, yguidefontsize=14, titlefontsize = 12)
p = []
for j in [2,3,4,5,6,7]
	quant999 = quantile(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),[0.0005,1-0.0005])
	βgrid = range(quant999[1],quant999[2], length = 1000)
	p_tmp = plot(βgrid, pdf(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),βgrid), 
		yaxis = false, yticks = [], title = Xnames[j])
	push!(p,p_tmp)
end
plot(size = (600,800), p..., layout = (3,2), margin = 0mm)
savefig(figFolder*"SalariesMargPost.pdf")

# Simulate from posterior κ₀ = 1

# Assistant professors
regLine = zeros(nSim, length(xstar))
for i = 1:nSim
    regLine[i,:] = exp.(Xstar*βsim[i,1:3])./1000
end
regLineMean = mean(regLine, dims = 1)[:]
regLineQuantLow = [quantile(regLine[:,j],0.025) for j in 1:length(xstar)]
regLineQuantHigh = [quantile(regLine[:,j],0.975) for j in 1:length(xstar)]
plot(xstar, regLineMean, color = colors[2], ribbon=(regLineMean-regLineQuantLow,regLineQuantHigh-regLineMean), 
    fillcolor = colors[1], fillalpha=0.4, label = "assistant professor", ylabel = "salary (in \$1000)", 
    xlabel = "years since PhD (normalized)")

# Associate professors
regLine = zeros(nSim, length(xstar))
for i = 1:nSim
    regLine[i,:] = exp.(Xstar*βsim[i,1:3] .+ βsim[i,4] )./1000
end
regLineMean = mean(regLine, dims = 1)[:]
regLineQuantLow = [quantile(regLine[:,j],0.025) for j in 1:length(xstar)]
regLineQuantHigh = [quantile(regLine[:,j],0.975) for j in 1:length(xstar)]
#plot!(xstar, regLineMean, color = colors[10], label = "associate professor")
#plot!(xstar, regLineQuantLow, color = colors[9], linestyle = :dash, lw = 2, label = "")
#plot!(xstar, regLineQuantHigh, color = colors[9], linestyle = :dash, lw = 2, label = "")

plot!(xstar, regLineMean, color = colors[10], ribbon=(regLineMean-regLineQuantLow,regLineQuantHigh-regLineMean), 
    fillcolor = colors[9], fillalpha=0.4, label = "associate professor", legend = :topleft)

# Full professors
regLine = zeros(nSim, length(xstar))
for i = 1:nSim
    regLine[i,:] = exp.(Xstar*βsim[i,1:3] .+ βsim[i,5])./1000
end
regLineMean = mean(regLine, dims = 1)[:]
regLineQuantLow = [quantile(regLine[:,j],0.025) for j in 1:length(xstar)]
regLineQuantHigh = [quantile(regLine[:,j],0.975) for j in 1:length(xstar)]
plot!(xstar, regLineMean, color = colors[6], ribbon=(regLineMean-regLineQuantLow,regLineQuantHigh-regLineMean), 
    fillcolor = colors[5], fillalpha=0.4, label = "full professor", legend = :topleft)

savefig(figFolder*"SalariesSimPostCurvesKappa1.pdf")


# Comparing carefully elicitated prior to unit information prior

βhat = X\y

# Careful prior
μ₀ = [log(70000), 2, -1.5, 0, 0, 0, 0]
ν₀ = 10
σ²₀ = 0.3^2
κ₀ = 1
Ω₀ = κ₀*(X'*X)
nSim = 10000
μₙ, Ωₙ, νₙ, σ²ₙ, βsim1, σ²sim1 = BayesLinReg(y, X, μ₀, Ω₀, ν₀, σ²₀, nSim)
covPost = σ²ₙ*inv(Ωₙ)
stdPost = .√((νₙ/(νₙ-2))diag(covPost))

# Marginal posterior densities - compare priors
gr(legendfontsize=8, xtickfontsize=8, ytickfontsize=8, xguidefontsize=14, yguidefontsize=14, titlefontsize = 12)
p = []
for j in [2,3,4,5,6,7]
	quant999 = quantile(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),[0.0001,1-0.0001])
	βgrid = range(quant999[1],quant999[2], length = 1000)
	p_tmp = plot(βgrid, pdf(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),βgrid), label = L"\kappa_0=1.0",
        yaxis = false, yticks = [], title = Xnames[j])
	push!(p,p_tmp)
end

# unit info prior
μ₀ = [log(70000), 2, -1.5, 0, 0, 0, 0]
ν₀ = 10
σ²₀ = 0.3^2
κ₀ = 0.1
Ω₀ = κ₀*(X'*X)
nSim = 10000
μₙ, Ωₙ, νₙ, σ²ₙ, βsim2, σ²sim2 = BayesLinReg(y, X, μ₀, Ω₀, ν₀, σ²₀, nSim)
covPost = σ²ₙ*inv(Ωₙ)
stdPost = .√((νₙ/(νₙ-2))diag(covPost))

for (i,j) in enumerate([2,3,4,5,6,7])
	quant999 = quantile(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),[0.0001,1-0.0001])
	βgrid = range(quant999[1],quant999[2], length = 1000)
	plot!(p[i],βgrid, pdf(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),βgrid), label = L"\kappa_0=0.1",
        yaxis = false, yticks = [], title = Xnames[j], color = colors[4], legend = :topleft)
    scatter!([βhat[j]],[0], color = colors[6], markersize = 4, label = L"\mathrm{ML}")
end

p
plot(size = (600,800), p..., layout = (3,2), margin = 0mm)
savefig(figFolder*"SalariesMargPostComparePriors.pdf")