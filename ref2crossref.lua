-- ref2crossref.lua
-- In HTML output, resolve LaTeX \ref{lab}, \eqref{lab} and \pageref{lab}
-- written in .qmd files to the book's numbers, e.g. \eqref{eq:likelinormal}
-- -> (2.9). The label map is written from the book's .aux by `make resolve`
-- (the pre-render hook). Labels not in the book fall back to Quarto @lab.

local labels = nil

local function load_labels()
  if labels then return labels end
  labels = {}
  local dir = (quarto and quarto.project and quarto.project.directory) or "."
  local path = dir .. "/exercises/resolved-exercises/labels.json"
  local f = io.open(path, "r")
  if f then
    labels = pandoc.json.decode(f:read("a"), false)
    f:close()
  else
    io.stderr:write("ref2crossref.lua: " .. path .. " not found; run `make resolve`\n")
  end
  return labels
end

local function resolve(cmd, label)
  local entry = load_labels()[label]
  if entry == nil then
    return "@" .. label
  end
  if cmd == "eqref" then
    return "(" .. entry.number .. ")"
  elseif cmd == "pageref" then
    return entry.page
  end
  return entry.number
end

function RawInline(el)
  if FORMAT:match("html") and el.format == "tex" then
    -- Look for patterns like \ref{ex:foo}, \eqref{eq:foo}, \pageref{foo}
    local new = el.text:gsub("\\(%a+)%s*{(.-)}", function(cmd, label)
      if cmd == "ref" or cmd == "eqref" or cmd == "pageref" then
        return resolve(cmd, label)
      end
    end)
    if new ~= el.text then
      -- Parse as markdown so an unresolved @lab still becomes a Quarto ref
      return pandoc.read(new, "markdown").blocks[1].content
    end
  end
  return nil
end
