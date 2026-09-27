-- rawtex.lua
-- In HTML output, raw LaTeX that pandoc can't read as markdown is silently
-- dropped. Convert the safe cases from the book's exercise .tex files with
-- pandoc's LaTeX reader: tabular tables and \textit/\textbf/\emph.
-- Other raw blocks (\input lists, marginfigure, tcolorbox, ...) are left alone.

local inline_cmds = { textit = true, textbf = true, emph = true }

function RawBlock(el)
  if FORMAT:match("html") and el.format == "tex"
      and el.text:match("^%s*\\begin{tabular}") then
    -- Sizing/spacing commands have no HTML meaning and confuse the reader
    local tex = el.text:gsub("\\scriptsize", ""):gsub("\\vspace%s*{[^}]*}", "")
    return pandoc.read(tex, "latex").blocks
  end
end

function RawInline(el)
  if FORMAT:match("html") and el.format == "tex" then
    local cmd = el.text:match("^\\(%a+)%s*{")
    if cmd and inline_cmds[cmd] then
      return pandoc.utils.blocks_to_inlines(pandoc.read(el.text, "latex").blocks)
    end
  end
end
