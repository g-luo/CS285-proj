def series_to_list(series):
  """
  Converts a pandas series to a one dimensional list
  """
  lst = series.tolist()
  if not isinstance(lst, list):
    lst = [lst]
  return lst
  