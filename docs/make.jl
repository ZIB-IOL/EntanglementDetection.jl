using Documenter
using EntanglementDetection

DocMeta.setdocmeta!(EntanglementDetection, :DocTestSetup, :(using EntanglementDetection); recursive = true)

generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/ZIB-IOL/EntanglementDetection.jl/blob/main/"
isdir(generated_path) || mkdir(generated_path)

open(joinpath(generated_path, "index.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)README.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

makedocs(;
    modules = [EntanglementDetection],
    authors = "ZIB AISST",
    repo = "https://github.com/ZIB-IOL/EntanglementDetection.jl/blob/{commit}{path}#{line}",
    sitename = "EntanglementDetection.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://zib-iol.github.io/EntanglementDetection.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md", "API reference" => "api.md"],
)

deploydocs(; repo = "github.com/ZIB-IOL/EntanglementDetection.jl", devbranch = "main", push_preview = true)
