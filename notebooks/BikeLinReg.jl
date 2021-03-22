# Linear regression daily bike share data
# For the book Bayesian Learning by Mattias Villani

using Plots, Distributions, LinearAlgebra, CSV, HTTP, DataFrames, Latexify, Measures
import ColorSchemes: Paired_12; colors = Paired_12[[1,2,7,8,3,4,5,6,9,10,11,12]]
using Flux: onehot, onehotbatch

bookFolder = "/home/mv/Dropbox/BayesBook/"
figFolder = bookFolder*"Figs/"
dataFolder = bookFolder*"Data/"
giturl = "https://github.com/mattiasvillani/BayesianLearningBook/"

include(bookFolder*"Code/Linreg.jl") # Functions for Bayesian linear regression
giturl = "https://github.com/mattiasvillani/BayesianLearningBook/raw/main/"
gr(legend = nothing, grid = false, color = colors[2], lw = 2, legendfontsize=12,
	xtickfontsize=12, ytickfontsize=12, xguidefontsize=14, yguidefontsize=14, 
	markersize = 4, markerstrokecolor = nothing)


# Read data from the book repository, and rename some variables so we understand what they measure.
url = giturl*"data/bikeShare/bikesday.csv"
bike = DataFrame(CSV.File(HTTP.get(url).body; header = true))
select!(bike, Not([:instant,:mnth,:workingday,:temp,:casual,:registered])) # remove some variables
rename!(bike, :cnt=>:nrides, :yr=>:year, :atemp=>:feeltemp, :weathersit=>:weather, :windspeed=>:wind)
bike.feeltempsqr = bike.feeltemp.^2

# Time series plot
plot(bike[!,:dteday],bike[!,:nrides], lw = 2, color = colors[2], 
	xlab = "Day", ylab = "Number of daily rides", legend = nothing)
savefig(figFolder*"biketimeseries.pdf")

# Scatterplot against feeltemp"
scatter(bike[!,:feeltemp], bike[!,:nrides], color = colors[6], xlab ="feeltemp", 	
	ylab = "Number of daily rides")
savefig(figFolder*"bikefeeltemp.pdf")

# The data look rather unpredictable with a large variance. 
# However, some of that variation will be explained other factors, for example 
# the weather (weather=1 is clear, weather=4 is heavy rain)
plot()
tmpcolors = colors[[2,1,4]]
for weather ∈ 1:3
	selDays = bike[!,:weather].==weather
	scatter!(bike[selDays,:feeltemp], bike[selDays,:nrides], color = tmpcolors[weather], 
		xlab ="feeltemp", ylab = "Number of rides", label = "weather = "*string(weather), 
		markerstrokecolor = tmpcolors[weather], legend = :topleft)
end
savefig(figFolder*"bikefeeltempbyweather.pdf")

plot()
tmpcolors = colors[[2,1,4,3]]
for seasonCount ∈ 1:4
	selDays = bike[!,:season].==seasonCount
	scatter!(bike[selDays,:feeltemp], bike[selDays,:nrides], color = tmpcolors[seasonCount], 
		xlab ="feeltemp", ylab = "Number of rides", label = "season = "*string(seasonCount), 
		markerstrokecolor = tmpcolors[seasonCount], legend = :topleft)
end
savefig(figFolder*"bikefeeltempbyseason.pdf")

plot()
tmpcolors = colors[[1,2,3,4]]
for yearCount ∈ [2011,2012]
	selDays = bike[!,:year].==(yearCount-2011)
	scatter!(bike[selDays,:feeltemp], bike[selDays,:nrides], color = tmpcolors[yearCount-2010], 
		xlab ="feeltemp", ylab = "Number of rides", label = "year = "*string(yearCount), 
		markerstrokecolor = tmpcolors[yearCount-2010], legend = :topleft)
end
savefig(figFolder*"bikefeeltempbyyear.pdf")

# Create binary dummy variables for categorical covariates (one-hot encoding)
Z = onehotbatch(bike[!,:season], [:1, :2, :3, :4])'
for k ∈ 2:size(Z,2)
	bike[!,Symbol("season",k)] = Z[:,k]
end
Z = onehotbatch(bike[!,:weather], [:1, :2, :3])'
for k ∈ 2:size(Z,2)
	bike[!,Symbol("weather",k)] = Z[:,k]
end
Z = onehotbatch(bike[!,:weekday], [:0, :1, :2, :3, :4, :5, :6])'
for k ∈ 1:6
	bike[!,Symbol("weekday",k)] = Z[:,k+1]
end

# Set up the response and covariate matrices
y = Vector(bike[!,:nrides])
X = bike[!,[:feeltemp,:hum,:wind,:year,:season2,:season3,:season4,:weather2,:weather3]]
# X = bike[[:feeltemp,:feeltempsqr,:hum,:wind,:year,:season2,:season3,:season4,:weather2,:weather3]]
X = [X bike[!,[:weekday1,:weekday2,:weekday3,:weekday4,:weekday5,:weekday6,:holiday]]]
#X = bike[[:feeltemp,:feeltempsqr]]

X = [ones(length(y)) X] # adding intercept as first column
rename!(X, :x1 => :intercept)
Xnames = names(X)
X = Matrix(X)
n, p = size(X)

# Prior
κ₀ = 1/n
μ₀ = zeros(p)
μ₀[1] = 1000
Ω₀ = κ₀*X'*X # κ₀*I(p) 
ν₀ = 5 # To guarantee a prior variance
σ²₀ = var(y)
nSim = 10000
μₙ, Ωₙ, νₙ, σ²ₙ, βsim, σ²sim = BayesLinReg(y, X, μ₀, Ω₀, ν₀, σ²₀, nSim)

stdPrior = .√((ν₀/(ν₀-2))*diag(σ²₀*inv(Ω₀))) # just checking
covPost = σ²ₙ*inv(Ωₙ)
stdPost = .√((νₙ/(νₙ-2))diag(covPost))


# Analytical marginal posteriors
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

gr(legendfontsize=8, xtickfontsize=8, ytickfontsize=8, xguidefontsize=14, yguidefontsize=14, titlefontsize = 12)
p = []
for j in [2,3,4,5,17]
	quant999 = quantile(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),[0.0005,1-0.0005])
	βgrid = range(quant999[1],quant999[2], length = 1000)
	p_tmp = plot(βgrid, pdf(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),βgrid), 
		yaxis = false, yticks = [], title = Xnames[j])
	push!(p,p_tmp)
end

# Seasons
tmpcolors = colors[[1,2,3,4,5,6]]
p_tmp = plot()
for (c,j) in enumerate(6:8)
	quant999 = quantile(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),[0.0005,1-0.0005])
	βgrid = range(quant999[1],quant999[2], length = 1000)
	p_tmp = plot!(βgrid, pdf(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),βgrid), label = "$(c+1)",
		yaxis = false, yticks = [], title = "season", color = tmpcolors[c], legend = :topleft)
end
push!(p,p_tmp)

# weather
p_tmp = plot()
for (c,j) in enumerate(9:10)
	quant999 = quantile(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),[0.0005,1-0.0005])
	βgrid = range(quant999[1],quant999[2], length = 1000)
	p_tmp = plot!(βgrid, pdf(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),βgrid), label = "$(c+1)",
		yaxis = false, yticks = [], title = "weather", color = tmpcolors[c],legend = :topleft)
end
push!(p,p_tmp)

# weekday
p_tmp = plot()
for (c,j) in enumerate(11:16)
	quant999 = quantile(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),[0.0005,1-0.0005])
	βgrid = range(quant999[1],quant999[2], length = 1000)
	p_tmp = plot!(βgrid, pdf(TDist(νₙ,μₙ[j],.√diag(covPost)[j]),βgrid), label = "$(c+1)", 
		yaxis = false, yticks = [], title = "weekday", color = tmpcolors[c],legend = :topleft)
end
push!(p,p_tmp)

plot(size = (600,800), p..., layout = (4,2), margin = 0mm)
savefig(figFolder*"bikemarginals.pdf")

# Bivariate posterior
subset = [2,3]
MVTpdf(β₁,β₂) = pdf(MvTDist(νₙ, μₙ[subset], covPost[subset,subset]),[β₁,β₂])
chisqVals = quantile.(Chisq(2),[0.1 0.25 0.5 0.7 0.9 0.975 0.99])
pdfAtContours = (1/√det(2*π*covPost))*exp.(-0.5*chisqVals)
quant999 = quantile(TDist(νₙ,μₙ[subset[1]],.√diag(covPost)[subset[1]]),[0.005,1-0.005])
β1grid = range(quant999[1],quant999[2], length = 1000)
quant999 = quantile(TDist(νₙ,μₙ[subset[2]],.√diag(covPost)[subset[2]]),[0.005,1-0.005])
β2grid = range(quant999[1],quant999[2], length = 1000)
contour(size=(450,450), β1grid, β2grid, MVTpdf, yaxis = true, color = :blues,
        xlabel = L"\beta_%$(subset[1])", ylabel = L"\beta_%$(subset[2])", lw = 2, levels = 10,
        margin = 5mm)
savefig(figFolder*"bikebivar1.pdf")

subset = [2,7]
MVTpdf(β₁,β₂) = pdf(MvTDist(νₙ, μₙ[subset], covPost[subset,subset]),[β₁,β₂])
chisqVals = quantile.(Chisq(2),[0.1 0.25 0.5 0.7 0.9 0.975 0.99])
pdfAtContours = (1/√det(2*π*covPost))*exp.(-0.5*chisqVals)
quant999 = quantile(TDist(νₙ,μₙ[subset[1]],.√diag(covPost)[subset[1]]),[0.005,1-0.005])
β1grid = range(quant999[1],quant999[2], length = 1000)
quant999 = quantile(TDist(νₙ,μₙ[subset[2]],.√diag(covPost)[subset[2]]),[0.005,1-0.005])
β2grid = range(quant999[1],quant999[2], length = 1000)
contour(size=(450,450), β1grid, β2grid, MVTpdf, yaxis = true, color = :blues,
        xlabel = L"\beta_%$(subset[1])", ylabel = L"\beta_%$(subset[2])", lw = 2, levels = 10,
        margin = 5mm)
savefig(figFolder*"bikebivar2.pdf")

# I will use:
# - the first 500 observations in time for learning the regression coefficients
# - the last 231 observations in time for evaluating the predictive performance of the model. 
bikeTrain = bike[1:500,:]
bikeTest = bike[501:end,:]

# Time series plot training/test
nTrain = 500
plot(bike[1:nTrain, :dteday],bike[1:nTrain,:nrides], lw = 2, color = colors[2], 
	xlab = "Day", ylab = "Number of daily rides", legend = nothing)
plot!(bike[nTrain+1:end,:dteday],bike[nTrain+1:end,:nrides], lw = 2, color = colors[4], 
	xlab = "Day", ylab = "Number of daily rides", legend = nothing)
savefig(figFolder*"biketimeseriessplitdata.pdf")





