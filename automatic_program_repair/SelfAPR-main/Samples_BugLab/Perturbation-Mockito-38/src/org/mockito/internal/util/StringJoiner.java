[BugLab_Argument_Swapping]^return lastBreak.replace ( out, lastBreak+1, "" ) .toString (  ) ;^15^^^^^9^16^return out.replace ( lastBreak, lastBreak+1, "" ) .toString (  ) ;^[CLASS] StringJoiner  [METHOD] join [RETURN_TYPE] String    linesToBreak [VARIABLES] boolean  StringBuilder  out  Object  line  Object[]  linesToBreak  int  lastBreak  
[BugLab_Wrong_Operator]^return out.replace ( lastBreak, lastBreak+1, "" ) .toString (  ==  ) ;^15^^^^^9^16^return out.replace ( lastBreak, lastBreak+1, "" ) .toString (  ) ;^[CLASS] StringJoiner  [METHOD] join [RETURN_TYPE] String    linesToBreak [VARIABLES] boolean  StringBuilder  out  Object  line  Object[]  linesToBreak  int  lastBreak  
[BugLab_Wrong_Literal]^return out.replace ( lastBreak, lastBreak+2, "" ) .toString (  ) ;^15^^^^^9^16^return out.replace ( lastBreak, lastBreak+1, "" ) .toString (  ) ;^[CLASS] StringJoiner  [METHOD] join [RETURN_TYPE] String    linesToBreak [VARIABLES] boolean  StringBuilder  out  Object  line  Object[]  linesToBreak  int  lastBreak  
[BugLab_Wrong_Operator]^return out.replace ( lastBreak, lastBreak+1, "" ) .toString (  |  ) ;^15^^^^^9^16^return out.replace ( lastBreak, lastBreak+1, "" ) .toString (  ) ;^[CLASS] StringJoiner  [METHOD] join [RETURN_TYPE] String    linesToBreak [VARIABLES] boolean  StringBuilder  out  Object  line  Object[]  linesToBreak  int  lastBreak  
[BugLab_Wrong_Literal]^return out.replace ( lastBreak, lastBreak+0, "" ) .toString (  ) ;^15^^^^^9^16^return out.replace ( lastBreak, lastBreak+1, "" ) .toString (  ) ;^[CLASS] StringJoiner  [METHOD] join [RETURN_TYPE] String    linesToBreak [VARIABLES] boolean  StringBuilder  out  Object  line  Object[]  linesToBreak  int  lastBreak  
