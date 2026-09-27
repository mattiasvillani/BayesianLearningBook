-- eqref.lua
-- Make Quarto equation references look like LaTeX \eqref: @eq-foo renders
-- as a link reading "(1)" instead of "Equation 1". Must run after Quarto's
-- crossref filter, i.e. listed after `quarto` under `filters:`.

function Link(el)
  if el.classes:includes("quarto-xref") and el.target:match("^#eq%-") then
    local num = pandoc.utils.stringify(el.content):match("([%w%.]+)$")
    if num then
      el.content = { pandoc.Str("(" .. num .. ")") }
      return el
    end
  end
end
