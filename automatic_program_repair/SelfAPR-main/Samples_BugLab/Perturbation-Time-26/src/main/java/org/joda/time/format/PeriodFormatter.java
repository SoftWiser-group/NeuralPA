[BugLab_Variable_Misuse]^iParser = iParser;^89^^^^^85^92^iParser = parser;^[CLASS] PeriodFormatter  [METHOD] <init> [RETURN_TYPE] PeriodParser)   PeriodPrinter printer PeriodParser parser [VARIABLES] Locale  iLocale  PeriodType  iParseType  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^iPrinter = iPrinter;^106^^^^^102^110^iPrinter = printer;^[CLASS] PeriodFormatter  [METHOD] <init> [RETURN_TYPE] PeriodType)   PeriodPrinter printer PeriodParser parser Locale locale PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^iParser = iParser;^107^^^^^102^110^iParser = parser;^[CLASS] PeriodFormatter  [METHOD] <init> [RETURN_TYPE] PeriodType)   PeriodPrinter printer PeriodParser parser Locale locale PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^iLocale = iLocale;^108^^^^^102^110^iLocale = locale;^[CLASS] PeriodFormatter  [METHOD] <init> [RETURN_TYPE] PeriodType)   PeriodPrinter printer PeriodParser parser Locale locale PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^iParseType = iParseType;^109^^^^^102^110^iParseType = type;^[CLASS] PeriodFormatter  [METHOD] <init> [RETURN_TYPE] PeriodType)   PeriodPrinter printer PeriodParser parser Locale locale PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return  ( printer != null ) ;^119^^^^^118^120^return  ( iPrinter != null ) ;^[CLASS] PeriodFormatter  [METHOD] isPrinter [RETURN_TYPE] boolean   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Wrong_Operator]^return  ( iPrinter == null ) ;^119^^^^^118^120^return  ( iPrinter != null ) ;^[CLASS] PeriodFormatter  [METHOD] isPrinter [RETURN_TYPE] boolean   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return printer;^128^^^^^127^129^return iPrinter;^[CLASS] PeriodFormatter  [METHOD] getPrinter [RETURN_TYPE] PeriodPrinter   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return  ( parser != null ) ;^137^^^^^136^138^return  ( iParser != null ) ;^[CLASS] PeriodFormatter  [METHOD] isParser [RETURN_TYPE] boolean   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Wrong_Operator]^return  ( iParser == null ) ;^137^^^^^136^138^return  ( iParser != null ) ;^[CLASS] PeriodFormatter  [METHOD] isParser [RETURN_TYPE] boolean   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return parser;^146^^^^^145^147^return iParser;^[CLASS] PeriodFormatter  [METHOD] getParser [RETURN_TYPE] PeriodParser   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^if  ( iLocale == getLocale (  )  ||  ( locale != null && locale.equals ( getLocale (  )  )  )  )  {^161^^^^^160^165^if  ( locale == getLocale (  )  ||  ( locale != null && locale.equals ( getLocale (  )  )  )  )  {^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Wrong_Operator]^if  ( locale == getLocale (  )  &&  ( locale != null && locale.equals ( getLocale (  )  )  )  )  {^161^^^^^160^165^if  ( locale == getLocale (  )  ||  ( locale != null && locale.equals ( getLocale (  )  )  )  )  {^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Wrong_Operator]^if  ( locale != getLocale (  )  ||  ( locale != null && locale.equals ( getLocale (  )  )  )  )  {^161^^^^^160^165^if  ( locale == getLocale (  )  ||  ( locale != null && locale.equals ( getLocale (  )  )  )  )  {^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Wrong_Operator]^if  ( locale == getLocale (  )  ||  ( locale != null || locale.equals ( getLocale (  )  )  )  )  {^161^^^^^160^165^if  ( locale == getLocale (  )  ||  ( locale != null && locale.equals ( getLocale (  )  )  )  )  {^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Wrong_Operator]^if  ( locale == getLocale (  )  ||  ( locale == null && locale.equals ( getLocale (  )  )  )  )  {^161^^^^^160^165^if  ( locale == getLocale (  )  ||  ( locale != null && locale.equals ( getLocale (  )  )  )  )  {^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return new PeriodFormatter ( printer, iParser, locale, iParseType ) ;^164^^^^^160^165^return new PeriodFormatter ( iPrinter, iParser, locale, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return new PeriodFormatter ( iPrinter, parser, locale, iParseType ) ;^164^^^^^160^165^return new PeriodFormatter ( iPrinter, iParser, locale, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return new PeriodFormatter ( iPrinter, iParser, locale, type ) ;^164^^^^^160^165^return new PeriodFormatter ( iPrinter, iParser, locale, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^return new PeriodFormatter ( iPrinter, iParser, iParseType, locale ) ;^164^^^^^160^165^return new PeriodFormatter ( iPrinter, iParser, locale, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^return new PeriodFormatter ( iParser, iPrinter, locale, iParseType ) ;^164^^^^^160^165^return new PeriodFormatter ( iPrinter, iParser, locale, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^return new PeriodFormatter ( iPrinter, iParseType, locale, iParser ) ;^164^^^^^160^165^return new PeriodFormatter ( iPrinter, iParser, locale, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return new PeriodFormatter ( iPrinter, iParser, iLocale, iParseType ) ;^164^^^^^160^165^return new PeriodFormatter ( iPrinter, iParser, locale, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^return new PeriodFormatter ( locale, iParser, iPrinter, iParseType ) ;^164^^^^^160^165^return new PeriodFormatter ( iPrinter, iParser, locale, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^return new PeriodFormatter ( iParseType, iParser, locale, iPrinter ) ;^164^^^^^160^165^return new PeriodFormatter ( iPrinter, iParser, locale, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^return new PeriodFormatter ( iPrinter, locale, iParser, iParseType ) ;^164^^^^^160^165^return new PeriodFormatter ( iPrinter, iParser, locale, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] withLocale [RETURN_TYPE] PeriodFormatter   Locale locale [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return locale;^173^^^^^172^174^return iLocale;^[CLASS] PeriodFormatter  [METHOD] getLocale [RETURN_TYPE] Locale   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^if  ( iParseType == type )  {^187^^^^^186^191^if  ( type == iParseType )  {^[CLASS] PeriodFormatter  [METHOD] withParseType [RETURN_TYPE] PeriodFormatter   PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Wrong_Operator]^if  ( type != iParseType )  {^187^^^^^186^191^if  ( type == iParseType )  {^[CLASS] PeriodFormatter  [METHOD] withParseType [RETURN_TYPE] PeriodFormatter   PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return new PeriodFormatter ( iPrinter, iParser, iLocale, iParseType ) ;^190^^^^^186^191^return new PeriodFormatter ( iPrinter, iParser, iLocale, type ) ;^[CLASS] PeriodFormatter  [METHOD] withParseType [RETURN_TYPE] PeriodFormatter   PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return new PeriodFormatter ( printer, iParser, iLocale, type ) ;^190^^^^^186^191^return new PeriodFormatter ( iPrinter, iParser, iLocale, type ) ;^[CLASS] PeriodFormatter  [METHOD] withParseType [RETURN_TYPE] PeriodFormatter   PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return new PeriodFormatter ( iPrinter, parser, iLocale, type ) ;^190^^^^^186^191^return new PeriodFormatter ( iPrinter, iParser, iLocale, type ) ;^[CLASS] PeriodFormatter  [METHOD] withParseType [RETURN_TYPE] PeriodFormatter   PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return new PeriodFormatter ( iPrinter, iParser, locale, type ) ;^190^^^^^186^191^return new PeriodFormatter ( iPrinter, iParser, iLocale, type ) ;^[CLASS] PeriodFormatter  [METHOD] withParseType [RETURN_TYPE] PeriodFormatter   PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^return new PeriodFormatter ( iPrinter, type, iLocale, iParser ) ;^190^^^^^186^191^return new PeriodFormatter ( iPrinter, iParser, iLocale, type ) ;^[CLASS] PeriodFormatter  [METHOD] withParseType [RETURN_TYPE] PeriodFormatter   PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^return new PeriodFormatter ( iParser, iPrinter, iLocale, type ) ;^190^^^^^186^191^return new PeriodFormatter ( iPrinter, iParser, iLocale, type ) ;^[CLASS] PeriodFormatter  [METHOD] withParseType [RETURN_TYPE] PeriodFormatter   PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^return new PeriodFormatter ( iPrinter, iLocale, iParser, type ) ;^190^^^^^186^191^return new PeriodFormatter ( iPrinter, iParser, iLocale, type ) ;^[CLASS] PeriodFormatter  [METHOD] withParseType [RETURN_TYPE] PeriodFormatter   PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^return new PeriodFormatter ( iLocale, iParser, iPrinter, type ) ;^190^^^^^186^191^return new PeriodFormatter ( iPrinter, iParser, iLocale, type ) ;^[CLASS] PeriodFormatter  [METHOD] withParseType [RETURN_TYPE] PeriodFormatter   PeriodType type [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return type;^199^^^^^198^200^return iParseType;^[CLASS] PeriodFormatter  [METHOD] getParseType [RETURN_TYPE] PeriodType   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^getPrinter (  ) .printTo ( buf, period, locale ) ;^213^^^^^209^214^getPrinter (  ) .printTo ( buf, period, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] printTo [RETURN_TYPE] void   StringBuffer buf ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  StringBuffer  buf  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^getPrinter (  ) .printTo ( period, buf, iLocale ) ;^213^^^^^209^214^getPrinter (  ) .printTo ( buf, period, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] printTo [RETURN_TYPE] void   StringBuffer buf ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  StringBuffer  buf  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^getPrinter (  ) .printTo ( buf, iLocale, period ) ;^213^^^^^209^214^getPrinter (  ) .printTo ( buf, period, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] printTo [RETURN_TYPE] void   StringBuffer buf ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  StringBuffer  buf  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^getPrinter (  ) .printTo ( out, period, locale ) ;^226^^^^^222^227^getPrinter (  ) .printTo ( out, period, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] printTo [RETURN_TYPE] void   Writer out ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  Writer  out  
[BugLab_Argument_Swapping]^getPrinter (  ) .printTo ( period, out, iLocale ) ;^226^^^^^222^227^getPrinter (  ) .printTo ( out, period, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] printTo [RETURN_TYPE] void   Writer out ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  Writer  out  
[BugLab_Argument_Swapping]^getPrinter (  ) .printTo ( out, iLocale, period ) ;^226^^^^^222^227^getPrinter (  ) .printTo ( out, period, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] printTo [RETURN_TYPE] void   Writer out ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  Writer  out  
[BugLab_Variable_Misuse]^StringBuffer buf = new StringBuffer ( iPrinter.calculatePrintedLength ( period, iLocale )  ) ;^240^^^^^235^243^StringBuffer buf = new StringBuffer ( printer.calculatePrintedLength ( period, iLocale )  ) ;^[CLASS] PeriodFormatter  [METHOD] print [RETURN_TYPE] String   ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  StringBuffer  buf  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^StringBuffer buf = new StringBuffer ( printer.calculatePrintedLength ( period, locale )  ) ;^240^^^^^235^243^StringBuffer buf = new StringBuffer ( printer.calculatePrintedLength ( period, iLocale )  ) ;^[CLASS] PeriodFormatter  [METHOD] print [RETURN_TYPE] String   ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  StringBuffer  buf  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^StringBuffer buf = new StringBuffer ( period.calculatePrintedLength ( printer, iLocale )  ) ;^240^^^^^235^243^StringBuffer buf = new StringBuffer ( printer.calculatePrintedLength ( period, iLocale )  ) ;^[CLASS] PeriodFormatter  [METHOD] print [RETURN_TYPE] String   ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  StringBuffer  buf  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^StringBuffer buf = new StringBuffer ( printer.calculatePrintedLength ( iLocale, period )  ) ;^240^^^^^235^243^StringBuffer buf = new StringBuffer ( printer.calculatePrintedLength ( period, iLocale )  ) ;^[CLASS] PeriodFormatter  [METHOD] print [RETURN_TYPE] String   ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  StringBuffer  buf  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^StringBuffer buf = new StringBuffer ( iLocale.calculatePrintedLength ( period, printer )  ) ;^240^^^^^235^243^StringBuffer buf = new StringBuffer ( printer.calculatePrintedLength ( period, iLocale )  ) ;^[CLASS] PeriodFormatter  [METHOD] print [RETURN_TYPE] String   ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  StringBuffer  buf  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^printer.printTo ( buf, period, locale ) ;^241^^^^^235^243^printer.printTo ( buf, period, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] print [RETURN_TYPE] String   ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  StringBuffer  buf  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Argument_Swapping]^printer.printTo ( buf, iLocale, period ) ;^241^^^^^235^243^printer.printTo ( buf, period, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] print [RETURN_TYPE] String   ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  StringBuffer  buf  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^if  ( printer == null )  {^251^^^^^250^254^if  ( iPrinter == null )  {^[CLASS] PeriodFormatter  [METHOD] checkPrinter [RETURN_TYPE] void   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Wrong_Operator]^if  ( iPrinter != null )  {^251^^^^^250^254^if  ( iPrinter == null )  {^[CLASS] PeriodFormatter  [METHOD] checkPrinter [RETURN_TYPE] void   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Wrong_Operator]^if  ( period != null )  {^262^^^^^261^265^if  ( period == null )  {^[CLASS] PeriodFormatter  [METHOD] checkPeriod [RETURN_TYPE] void   ReadablePeriod period [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  ReadablePeriod  period  PeriodPrinter  iPrinter  printer  
[BugLab_Variable_Misuse]^return getParser (  ) .parseInto ( period, text, position, locale ) ;^291^^^^^287^292^return getParser (  ) .parseInto ( period, text, position, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseInto [RETURN_TYPE] int   ReadWritablePeriod period String text int position [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  ReadWritablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  position  
[BugLab_Argument_Swapping]^return getParser (  ) .parseInto ( position, text, period, iLocale ) ;^291^^^^^287^292^return getParser (  ) .parseInto ( period, text, position, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseInto [RETURN_TYPE] int   ReadWritablePeriod period String text int position [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  ReadWritablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  position  
[BugLab_Argument_Swapping]^return getParser (  ) .parseInto ( period, position, text, iLocale ) ;^291^^^^^287^292^return getParser (  ) .parseInto ( period, text, position, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseInto [RETURN_TYPE] int   ReadWritablePeriod period String text int position [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  ReadWritablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  position  
[BugLab_Argument_Swapping]^return getParser (  ) .parseInto ( iLocale, text, position, period ) ;^291^^^^^287^292^return getParser (  ) .parseInto ( period, text, position, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseInto [RETURN_TYPE] int   ReadWritablePeriod period String text int position [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  ReadWritablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  position  
[BugLab_Argument_Swapping]^return getParser (  ) .parseInto ( text, period, position, iLocale ) ;^291^^^^^287^292^return getParser (  ) .parseInto ( period, text, position, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseInto [RETURN_TYPE] int   ReadWritablePeriod period String text int position [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  ReadWritablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  position  
[BugLab_Argument_Swapping]^return getParser (  ) .parseInto ( period, iLocale, position, text ) ;^291^^^^^287^292^return getParser (  ) .parseInto ( period, text, position, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseInto [RETURN_TYPE] int   ReadWritablePeriod period String text int position [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  ReadWritablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  position  
[BugLab_Variable_Misuse]^MutablePeriod period = new MutablePeriod ( 0, type ) ;^317^^^^^314^327^MutablePeriod period = new MutablePeriod ( 0, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Wrong_Literal]^MutablePeriod period = new MutablePeriod ( newPos, iParseType ) ;^317^^^^^314^327^MutablePeriod period = new MutablePeriod ( 0, iParseType ) ;^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Variable_Misuse]^int newPos = getParser (  ) .parseInto ( period, text, 0, locale ) ;^318^^^^^314^327^int newPos = getParser (  ) .parseInto ( period, text, 0, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Argument_Swapping]^int newPos = getParser (  ) .parseInto ( text, period, 0, iLocale ) ;^318^^^^^314^327^int newPos = getParser (  ) .parseInto ( period, text, 0, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Argument_Swapping]^int newPos = getParser (  ) .parseInto ( period, iLocale, 0, text ) ;^318^^^^^314^327^int newPos = getParser (  ) .parseInto ( period, text, 0, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Wrong_Literal]^int newPos = getParser (  ) .parseInto ( period, text, -1, iLocale ) ;^318^^^^^314^327^int newPos = getParser (  ) .parseInto ( period, text, 0, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Wrong_Literal]^int newPos = getParser (  ) .parseInto ( period, text, 1, iLocale ) ;^318^^^^^314^327^int newPos = getParser (  ) .parseInto ( period, text, 0, iLocale ) ;^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Wrong_Operator]^if  ( newPos < 0 )  {^319^^^^^314^327^if  ( newPos >= 0 )  {^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Wrong_Operator]^if  ( newPos == 0 )  {^319^^^^^314^327^if  ( newPos >= 0 )  {^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Wrong_Literal]^if  ( newPos >= newPos )  {^319^^^^^314^327^if  ( newPos >= 0 )  {^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Argument_Swapping]^if  ( text >= newPos.length (  )  )  {^320^^^^^314^327^if  ( newPos >= text.length (  )  )  {^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Wrong_Operator]^if  ( newPos > text.length (  )  )  {^320^^^^^314^327^if  ( newPos >= text.length (  )  )  {^[CLASS] PeriodFormatter  [METHOD] parseMutablePeriod [RETURN_TYPE] MutablePeriod   String text [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  MutablePeriod  period  String  text  PeriodPrinter  iPrinter  printer  int  newPos  
[BugLab_Variable_Misuse]^if  ( parser == null )  {^335^^^^^334^338^if  ( iParser == null )  {^[CLASS] PeriodFormatter  [METHOD] checkParser [RETURN_TYPE] void   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  
[BugLab_Wrong_Operator]^if  ( iParser != null )  {^335^^^^^334^338^if  ( iParser == null )  {^[CLASS] PeriodFormatter  [METHOD] checkParser [RETURN_TYPE] void   [VARIABLES] Locale  iLocale  locale  PeriodType  iParseType  type  boolean  PeriodParser  iParser  parser  PeriodPrinter  iPrinter  printer  