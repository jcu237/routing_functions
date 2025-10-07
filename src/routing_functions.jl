struct routing_function
    f::Expression               # numerator expression
    g::Expression               # denominator expression
                                    # computed from f and c
    c::Vector{Float64}          # center for numerator
                                    # can be specified, otherwise chosen randomly
    d::Int64                    # degree of num
    eval::Function              # function to HC.evaluate routing function num
    grad::Vector{Expression}    # gradient of routing function
    eval_grad::Function         # HC.evaluates gradient at a point
    vars::Vector{Variable}      # vector of variables.
                                    # if f has all variables present, 
                                    # this will be computed automatically

    function routing_function(f::Expression, vars::Vector{Variable}, c::Vector{Float64})

        g = sum([(vars[i] - c[i])^2 for i in eachindex(vars)]) + 1
        d = degree(f) ÷ 2 + 1
        
        function eval_fun(P::Vector{Float64})
            if length(P) != length(vars)
                error("length of point must match number of variables")
            else
                HC.evaluate(f/g^d, vars => P)
            end
        end

        grad = HC.differentiate(f/g^d, vars)

        function eval_grad(P::Vector{Float64})
            if length(P) != length(vars)
                error("length of point must match number of variables")
            else
                HC.evaluate(grad, vars => P)
            end
        end

        new(f, g, c, d, eval_fun, grad, eval_grad, vars)
    end

    function routing_function(f::Expression, c::Vector{Float64})
        vars = variables(f)
        return routing_function(f, vars, c)
    end

    function routing_function(f::Expression, vars::Vector{Variable})
        c = rand(length(vars))
        return routing_function(f, vars, c)
    end

    function routing_function(f::Expression)
        vars = variables(f)
        c = rand(length(vars))
        return routing_function(f, vars, c)
    end
end

function routing_system(
    r::routing_function, 
    G::Vector{Expression}
    )::System
    
    @var λ[1:length(G)]
    ∇f = HC.differentiate(r.f, r.vars)
    ∇g = HC.differentiate(r.g, r.vars)
    ∇r_num = ∇f * r.g - r.d * r.f * ∇g
    eqns = vcat(∇r_num - (r.g)^(r.d + 1) * transpose(HC.differentiate(G, r.vars)) * λ, G)
    return System(eqns; variables=vcat(r.vars, λ))
end

function routing_points(
    r::routing_function, 
    G::Vector{Expression};
    all_vars::Bool = false,
    zero_tol::Float64 = 1e-5
    )::Vector{Vector{Float64}}
    
    sys = routing_system(r, G)
    res = HC.solve(sys)
    if all_vars
        return [p for p in HC.real_solutions(res) if abs(HC.evaluate(r.f, r.vars => p)) > zero_tol]
    else
        return [p[1:length(r.vars)] for p in HC.real_solutions(res) if abs(HC.evaluate(r.f, r.vars => p[1:length(r.vars)])) > zero_tol]
    end
end