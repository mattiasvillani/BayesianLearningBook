-- rawtex.lua
-- In HTML output, raw LaTeX that pandoc can't read as markdown is silently
-- dropped. Convert the safe cases from the book's exercise .tex files with
-- pandoc's LaTeX reader: tabular tables and \textit/\textbf/\emph.
-- Other raw blocks (marginfigure, tcolorbox, ...) are left alone.
-- \numprint{10000} becomes 10,000 in all formats, as in the book.
-- In PDF output, drop the book's \input lists of exercise parts: the paths
-- only exist in the book, and the .qmd files include each part themselves.

local inline_cmds = { textit = true, textbf = true, emph = true }

function RawBlock(el)
  if FORMAT:match("html") and el.format == "tex"
      and el.text:match("^%s*\\begin{tabular}") then
    -- Sizing/spacing commands have no HTML meaning and confuse the reader
    local tex = el.text:gsub("\\scriptsize", ""):gsub("\\vspace%s*{[^}]*}", "")
    return pandoc.read(tex, "latex").blocks
  end
  if not FORMAT:match("html") and el.format == "tex"
      and el.text:match("\\input%s*{exercises/") then
    return {}
  end
end

function RawInline(el)
  if el.format == "tex" then
    local num = el.text:match("^\\numprint%s*{%s*(%d+)%s*}$")
    if num then
      local formatted = num:reverse():gsub("(%d%d%d)", "%1,"):reverse():gsub("^,", "")
      return pandoc.Str(formatted)
    end
  end
  if FORMAT:match("html") and el.format == "tex" then
    local cmd = el.text:match("^\\(%a+)%s*{")
    if cmd and inline_cmds[cmd] then
      return pandoc.utils.blocks_to_inlines(pandoc.read(el.text, "latex").blocks)
    end
  end
end
