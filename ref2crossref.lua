-- ref2crossref.lua
-- Convert LaTeX-style \ref{ex:foo} into Quarto @ex-foo when rendering HTML

function RawInline(el)
  if FORMAT:match("html") and el.format == "tex" then
    -- Look for patterns like \ref{ex:foo}
    local new = el.text:gsub("\\ref%s*{(.-)}", "@%1")
    if new ~= el.text then
      return pandoc.RawInline("markdown", new)
    end
  end
  return nil
end
