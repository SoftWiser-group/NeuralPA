[BugLab_Argument_Swapping]^this ( spec, gson, null ) ;^49^^^^^48^50^this ( gson, spec, null ) ;^[CLASS] ResponseReceiver  [METHOD] <init> [RETURN_TYPE] ResponseSpec)   Gson gson ResponseSpec spec [VARIABLES] Gson  gson  Level  logLevel  boolean  Logger  logger  ResponseSpec  spec  
[BugLab_Wrong_Operator]^this.logger = logLevel != null ? null : Logger.getLogger ( ResponseReceiver.class.getName (  )  ) ;^54^^^^^51^56^this.logger = logLevel == null ? null : Logger.getLogger ( ResponseReceiver.class.getName (  )  ) ;^[CLASS] ResponseReceiver  [METHOD] <init> [RETURN_TYPE] Level)   Gson gson ResponseSpec spec Level logLevel [VARIABLES] Gson  gson  Level  logLevel  boolean  Logger  logger  ResponseSpec  spec  
[BugLab_Argument_Swapping]^return new WebServiceResponse ( responseBody, responseParams ) ;^65^^^^^58^69^return new WebServiceResponse ( responseParams, responseBody ) ;^[CLASS] ResponseReceiver  [METHOD] receive [RETURN_TYPE] WebServiceResponse   HttpURLConnection conn [VARIABLES] HttpURLConnection  conn  ResponseBodySpec  bodySpec  boolean  HeaderMap  responseParams  Gson  gson  Level  logLevel  IOException  e  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramSpec  ResponseBody  responseBody  
[BugLab_Argument_Swapping]^HeaderMap responseParams = readResponseHeaders ( paramSpec, conn ) ;^63^^^^^58^69^HeaderMap responseParams = readResponseHeaders ( conn, paramSpec ) ;^[CLASS] ResponseReceiver  [METHOD] receive [RETURN_TYPE] WebServiceResponse   HttpURLConnection conn [VARIABLES] HttpURLConnection  conn  ResponseBodySpec  bodySpec  boolean  HeaderMap  responseParams  Gson  gson  Level  logLevel  IOException  e  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramSpec  ResponseBody  responseBody  
[BugLab_Argument_Swapping]^ResponseBody responseBody = readResponseBody ( bodySpec, conn ) ;^64^^^^^58^69^ResponseBody responseBody = readResponseBody ( conn, bodySpec ) ;^[CLASS] ResponseReceiver  [METHOD] receive [RETURN_TYPE] WebServiceResponse   HttpURLConnection conn [VARIABLES] HttpURLConnection  conn  ResponseBodySpec  bodySpec  boolean  HeaderMap  responseParams  Gson  gson  Level  logLevel  IOException  e  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramSpec  ResponseBody  responseBody  
[BugLab_Variable_Misuse]^if  ( paramName != null )  {^76^^^^^71^86^if  ( json != null )  {^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Wrong_Operator]^if  ( json == null )  {^76^^^^^71^86^if  ( json != null )  {^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Wrong_Operator]^if  ( logger == null )  {^77^^^^^71^86^if  ( logger != null )  {^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Variable_Misuse]^logger.log ( logLevel, String.format ( "Response Header: %s:%s\n", json, json )  ) ;^78^^^^^71^86^logger.log ( logLevel, String.format ( "Response Header: %s:%s\n", paramName, json )  ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Variable_Misuse]^logger.log ( logLevel, String.format ( "Response Header: %s:%s\n", paramName, paramName )  ) ;^78^^^^^71^86^logger.log ( logLevel, String.format ( "Response Header: %s:%s\n", paramName, json )  ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Argument_Swapping]^logger.log ( logLevel, String.format ( "Response Header: %s:%s\n", json, paramName )  ) ;^78^^^^^71^86^logger.log ( logLevel, String.format ( "Response Header: %s:%s\n", paramName, json )  ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Variable_Misuse]^Type typeOfT = paramsSpec.getTypeFor ( json ) ;^80^^^^^71^86^Type typeOfT = paramsSpec.getTypeFor ( paramName ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Argument_Swapping]^Type typeOfT = paramName.getTypeFor ( paramsSpec ) ;^80^^^^^71^86^Type typeOfT = paramsSpec.getTypeFor ( paramName ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Variable_Misuse]^Object value = gson.fromJson ( paramName, typeOfT ) ;^81^^^^^71^86^Object value = gson.fromJson ( json, typeOfT ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Argument_Swapping]^Object value = gson.fromJson ( typeOfT, json ) ;^81^^^^^71^86^Object value = gson.fromJson ( json, typeOfT ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Argument_Swapping]^Object value = typeOfT.fromJson ( json, gson ) ;^81^^^^^71^86^Object value = gson.fromJson ( json, typeOfT ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Argument_Swapping]^logger.log ( json, String.format ( "Response Header: %s:%s\n", paramName, logLevel )  ) ;^78^^^^^71^86^logger.log ( logLevel, String.format ( "Response Header: %s:%s\n", paramName, json )  ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Argument_Swapping]^Object value = json.fromJson ( gson, typeOfT ) ;^81^^^^^71^86^Object value = gson.fromJson ( json, typeOfT ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Variable_Misuse]^paramsBuilder.put ( json, value, typeOfT ) ;^82^^^^^71^86^paramsBuilder.put ( paramName, value, typeOfT ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Argument_Swapping]^paramsBuilder.put ( typeOfT, value, paramName ) ;^82^^^^^71^86^paramsBuilder.put ( paramName, value, typeOfT ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Argument_Swapping]^String json = paramName.getHeaderField ( conn ) ;^75^^^^^71^86^String json = conn.getHeaderField ( paramName ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Variable_Misuse]^String json = conn.getHeaderField ( json ) ;^75^^^^^71^86^String json = conn.getHeaderField ( paramName ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Argument_Swapping]^paramsBuilder.put ( value, paramName, typeOfT ) ;^82^^^^^71^86^paramsBuilder.put ( paramName, value, typeOfT ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Argument_Swapping]^paramsBuilder.put ( paramName, typeOfT, value ) ;^82^^^^^71^86^paramsBuilder.put ( paramName, value, typeOfT ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseHeaders [RETURN_TYPE] HeaderMap   HttpURLConnection conn HeaderMapSpec paramsSpec [VARIABLES] Entry  entry  Type  typeOfT  HttpURLConnection  conn  boolean  Builder  paramsBuilder  Gson  gson  Level  logLevel  Object  value  String  json  paramName  Logger  logger  ResponseSpec  spec  HeaderMapSpec  paramsSpec  
[BugLab_Wrong_Operator]^if  ( bodySpec.size (  )  >= 0 )  {^90^^^^^88^98^if  ( bodySpec.size (  )  == 0 )  {^[CLASS] ResponseReceiver  [METHOD] readResponseBody [RETURN_TYPE] ResponseBody   HttpURLConnection conn ResponseBodySpec bodySpec [VARIABLES] HttpURLConnection  conn  ResponseBodySpec  bodySpec  boolean  Gson  gson  Reader  reader  Level  logLevel  String  connContentType  Logger  logger  ResponseSpec  spec  ResponseBody  body  
[BugLab_Wrong_Literal]^if  ( bodySpec.size (  )  ==  )  {^90^^^^^88^98^if  ( bodySpec.size (  )  == 0 )  {^[CLASS] ResponseReceiver  [METHOD] readResponseBody [RETURN_TYPE] ResponseBody   HttpURLConnection conn ResponseBodySpec bodySpec [VARIABLES] HttpURLConnection  conn  ResponseBodySpec  bodySpec  boolean  Gson  gson  Reader  reader  Level  logLevel  String  connContentType  Logger  logger  ResponseSpec  spec  ResponseBody  body  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( conn.contains ( bodySpec.getContentType (  )  ) , connContentType ) ;^94^^^^^88^98^Preconditions.checkArgument ( connContentType.contains ( bodySpec.getContentType (  )  ) , conn ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseBody [RETURN_TYPE] ResponseBody   HttpURLConnection conn ResponseBodySpec bodySpec [VARIABLES] HttpURLConnection  conn  ResponseBodySpec  bodySpec  boolean  Gson  gson  Reader  reader  Level  logLevel  String  connContentType  Logger  logger  ResponseSpec  spec  ResponseBody  body  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( bodySpecContentType.contains ( conn.getContentType (  )  ) , conn ) ;^94^^^^^88^98^Preconditions.checkArgument ( connContentType.contains ( bodySpec.getContentType (  )  ) , conn ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseBody [RETURN_TYPE] ResponseBody   HttpURLConnection conn ResponseBodySpec bodySpec [VARIABLES] HttpURLConnection  conn  ResponseBodySpec  bodySpec  boolean  Gson  gson  Reader  reader  Level  logLevel  String  connContentType  Logger  logger  ResponseSpec  spec  ResponseBody  body  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( bodySpec.contains ( connContentType.getContentType (  )  ) , conn ) ;^94^^^^^88^98^Preconditions.checkArgument ( connContentType.contains ( bodySpec.getContentType (  )  ) , conn ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseBody [RETURN_TYPE] ResponseBody   HttpURLConnection conn ResponseBodySpec bodySpec [VARIABLES] HttpURLConnection  conn  ResponseBodySpec  bodySpec  boolean  Gson  gson  Reader  reader  Level  logLevel  String  connContentType  Logger  logger  ResponseSpec  spec  ResponseBody  body  
[BugLab_Argument_Swapping]^ResponseBody body = reader.fromJson ( gson, ResponseBody.class ) ;^96^^^^^88^98^ResponseBody body = gson.fromJson ( reader, ResponseBody.class ) ;^[CLASS] ResponseReceiver  [METHOD] readResponseBody [RETURN_TYPE] ResponseBody   HttpURLConnection conn ResponseBodySpec bodySpec [VARIABLES] HttpURLConnection  conn  ResponseBodySpec  bodySpec  boolean  Gson  gson  Reader  reader  Level  logLevel  String  connContentType  Logger  logger  ResponseSpec  spec  ResponseBody  body  
