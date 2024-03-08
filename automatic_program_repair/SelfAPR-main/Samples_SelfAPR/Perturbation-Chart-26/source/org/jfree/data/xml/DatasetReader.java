[P7_Replace_Invocation]^return readCategoryDatasetFromXML ( in ) ;^75^^^^^72^76^return readPieDatasetFromXML ( in ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   File file [VARIABLES] boolean  InputStream  in  File  file  
[P14_Delete_Statement]^^75^^^^^72^76^return readPieDatasetFromXML ( in ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   File file [VARIABLES] boolean  InputStream  in  File  file  
[P11_Insert_Donor_Statement]^CategoryDataset result = null;PieDataset result = null;^90^^^^^87^106^PieDataset result = null;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P7_Replace_Invocation]^SAXParserFactory factory = SAXParserFactory.newSAXParser (  ) ;^91^^^^^87^106^SAXParserFactory factory = SAXParserFactory.newInstance (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P11_Insert_Donor_Statement]^SAXParser parser = factory.newSAXParser (  ) ;SAXParserFactory factory = SAXParserFactory.newInstance (  ) ;^91^^^^^87^106^SAXParserFactory factory = SAXParserFactory.newInstance (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^91^^^^^87^106^SAXParserFactory factory = SAXParserFactory.newInstance (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P7_Replace_Invocation]^result = handler .PieDatasetHandler (  )  ;^96^^^^^87^106^result = handler.getDataset (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P8_Replace_Mix]^result =  null.getDataset (  ) ;^96^^^^^87^106^result = handler.getDataset (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P4_Replace_Constructor]^PieDatasetHandler handler = new CategoryDatasetHandler (  )  ;^94^^^^^87^106^PieDatasetHandler handler = new PieDatasetHandler (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P7_Replace_Invocation]^SAXParser parser = factory .newInstance (  )  ;^93^^^^^87^106^SAXParser parser = factory.newSAXParser (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P11_Insert_Donor_Statement]^SAXParserFactory factory = SAXParserFactory.newInstance (  ) ;SAXParser parser = factory.newSAXParser (  ) ;^93^^^^^87^106^SAXParser parser = factory.newSAXParser (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P11_Insert_Donor_Statement]^CategoryDatasetHandler handler = new CategoryDatasetHandler (  ) ;PieDatasetHandler handler = new PieDatasetHandler (  ) ;^94^^^^^87^106^PieDatasetHandler handler = new PieDatasetHandler (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^93^94^^^^87^106^SAXParser parser = factory.newSAXParser (  ) ; PieDatasetHandler handler = new PieDatasetHandler (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P5_Replace_Variable]^parser.parse (  handler ) ;^95^^^^^87^106^parser.parse ( in, handler ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P5_Replace_Variable]^parser.parse ( in ) ;^95^^^^^87^106^parser.parse ( in, handler ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P5_Replace_Variable]^parser.parse ( handler, in ) ;^95^^^^^87^106^parser.parse ( in, handler ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^95^96^^^^87^106^parser.parse ( in, handler ) ; result = handler.getDataset (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^96^^^^^87^106^result = handler.getDataset (  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^99^^^^^87^106^System.out.println ( e.getMessage (  )  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P11_Insert_Donor_Statement]^System.out.println ( e2.getMessage (  )  ) ;System.out.println ( e.getMessage (  )  ) ;^99^^^^^87^106^System.out.println ( e.getMessage (  )  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^102^^^^^87^106^System.out.println ( e2.getMessage (  )  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P11_Insert_Donor_Statement]^System.out.println ( e.getMessage (  )  ) ;System.out.println ( e2.getMessage (  )  ) ;^102^^^^^87^106^System.out.println ( e2.getMessage (  )  ) ;^[CLASS] DatasetReader  [METHOD] readPieDatasetFromXML [RETURN_TYPE] PieDataset   InputStream in [VARIABLES] boolean  PieDataset  result  InputStream  in  PieDatasetHandler  handler  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P7_Replace_Invocation]^return readPieDatasetFromXML ( in ) ;^120^^^^^117^121^return readCategoryDatasetFromXML ( in ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   File file [VARIABLES] boolean  InputStream  in  File  file  
[P14_Delete_Statement]^^120^^^^^117^121^return readCategoryDatasetFromXML ( in ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   File file [VARIABLES] boolean  InputStream  in  File  file  
[P11_Insert_Donor_Statement]^PieDataset result = null;CategoryDataset result = null;^135^^^^^132^152^CategoryDataset result = null;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P7_Replace_Invocation]^SAXParserFactory factory = SAXParserFactory.newSAXParser (  ) ;^137^^^^^132^152^SAXParserFactory factory = SAXParserFactory.newInstance (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P11_Insert_Donor_Statement]^SAXParser parser = factory.newSAXParser (  ) ;SAXParserFactory factory = SAXParserFactory.newInstance (  ) ;^137^^^^^132^152^SAXParserFactory factory = SAXParserFactory.newInstance (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^137^^^^^132^152^SAXParserFactory factory = SAXParserFactory.newInstance (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P8_Replace_Mix]^result =  null.getDataset (  ) ;^142^^^^^132^152^result = handler.getDataset (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P4_Replace_Constructor]^CategoryDatasetHandler handler = new PieDatasetHandler (  )  ;^140^^^^^132^152^CategoryDatasetHandler handler = new CategoryDatasetHandler (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P7_Replace_Invocation]^SAXParser parser = factory .newInstance (  )  ;^139^^^^^132^152^SAXParser parser = factory.newSAXParser (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P11_Insert_Donor_Statement]^SAXParserFactory factory = SAXParserFactory.newInstance (  ) ;SAXParser parser = factory.newSAXParser (  ) ;^139^^^^^132^152^SAXParser parser = factory.newSAXParser (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P11_Insert_Donor_Statement]^PieDatasetHandler handler = new PieDatasetHandler (  ) ;CategoryDatasetHandler handler = new CategoryDatasetHandler (  ) ;^140^^^^^132^152^CategoryDatasetHandler handler = new CategoryDatasetHandler (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^139^^^^^132^152^SAXParser parser = factory.newSAXParser (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P5_Replace_Variable]^parser.parse (  handler ) ;^141^^^^^132^152^parser.parse ( in, handler ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P5_Replace_Variable]^parser.parse ( in ) ;^141^^^^^132^152^parser.parse ( in, handler ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^141^^^^^132^152^parser.parse ( in, handler ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^142^^^^^132^152^result = handler.getDataset (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P5_Replace_Variable]^parser.parse ( handler, in ) ;^141^^^^^132^152^parser.parse ( in, handler ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^141^142^^^^132^152^parser.parse ( in, handler ) ; result = handler.getDataset (  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^145^^^^^132^152^System.out.println ( e.getMessage (  )  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P11_Insert_Donor_Statement]^System.out.println ( e2.getMessage (  )  ) ;System.out.println ( e.getMessage (  )  ) ;^145^^^^^132^152^System.out.println ( e.getMessage (  )  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P14_Delete_Statement]^^148^^^^^132^152^System.out.println ( e2.getMessage (  )  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  
[P11_Insert_Donor_Statement]^System.out.println ( e.getMessage (  )  ) ;System.out.println ( e2.getMessage (  )  ) ;^148^^^^^132^152^System.out.println ( e2.getMessage (  )  ) ;^[CLASS] DatasetReader  [METHOD] readCategoryDatasetFromXML [RETURN_TYPE] CategoryDataset   InputStream in [VARIABLES] boolean  CategoryDatasetHandler  handler  InputStream  in  CategoryDataset  result  SAXParser  parser  ParserConfigurationException  e2  SAXParserFactory  factory  SAXException  e  