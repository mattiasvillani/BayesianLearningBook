function RawInline(el)
  if FORMAT:match('html') and el.format:match('tex') then
    local content = el.text:match("\\texttt%s*{%s*(.-)%s*}")
    if content then
      return pandoc.Code(content)
    end
  end
  return nil
end
